# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from . import ex_vssd_cls_seg as mamba
from . import MSAA
from . import FMB
from . import DySample
from . import DSCB
from . import cga_block as cga
from . import DICAM
from . import DA_Block as DA
from . import DA_TRANSUNET_block

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu,
          "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(
            config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(
            config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(
            config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, backbone, in_channels=3):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (img_size[0] // 16 // grid_size[0],
                      img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * \
            (img_size[1] // patch_size_real[1])
        if self.config.preprocess == 'DICAM':
            self.preprocess = DICAM.DICAM()
        self.hybrid_model = backbone
        self.patch_embeddings = Conv2d(in_channels=1024,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.config.preprocess == 'DICAM':
            x = self.preprocess(x)
        x, features = self.hybrid_model(x)
        # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()

            query_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class MambaTransblock(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(
                self.hidden_size, self.hidden_size).t()

            query_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(
                weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            self.attention_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            if config.vit_post == 'mamba2' and i == config.transformer["num_layers"]-1:
                layer = MambaTransblock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, backbone):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size, backbone)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(
            embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SegmentationHead(nn.Sequential):

    def __init__(self, config, in_channels, out_channels, kernel_size=3, upsampling=1):
        if config.seghead == '':
            pass
        else:
            conv2d = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DASegmentationHead(nn.Sequential):

    def __init__(self, config, in_channels, out_channels, kernel_size=3, upsampling=1):
        if config.seghead == '':
            pass
        else:
            conv2d = DA_TRANSUNET_block.DANetHead(in_channels, out_channels)
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class RPLinearBlock(nn.Module):
    def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
        super(RPLinearBlock, self).__init__()
        # rpl 失败，换成原来的rp试试。为什么在这部分继续加模块就没用呢？
    #     self.branches = nn.ModuleList()
    #     for i, ratio in enumerate(ratios):
    #         conv = nn.Sequential(
    #             nn.Conv2d(input_chs, input_chs,
    #                       kernel_size=(1, 2 * i+1),
    #                       padding=(0, i)),
    #             nn.BatchNorm2d(input_chs, momentum=bn_momentum),
    #             nn.Conv2d(input_chs, int(input_chs * ratio),
    #                       kernel_size=(2 * i+1, 1),
    #                       padding=(i, 0)),
    #             nn.ReLU()
    #         )
    #         self.branches.append(conv)

    #     self.fuse_conv = nn.Sequential(  # + input_chs // 64
    #         nn.Conv2d(int(input_chs * sum(ratios)), input_chs,
    #                   kernel_size=1, stride=1, padding=0),
    #         nn.BatchNorm2d(input_chs, momentum=bn_momentum),
    #         nn.ReLU()
    #     )

    # def forward(self, x):
    #     branches = torch.cat([branch(x) for branch in self.branches], dim=1)
    #     output = self.fuse_conv(branches) + x
    #     return output
        self.branches = nn.ModuleList()
        for i, ratio in enumerate(ratios):
            conv = nn.Sequential(
                nn.Conv2d(input_chs, int(input_chs * ratio),
                          kernel_size=(2 * i + 1), stride=1, padding=i),
                nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
                nn.ReLU()
            )
            self.branches.append(conv)

        self.fuse_conv = nn.Sequential(  # + input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        output = self.fuse_conv(branches) + x
        return output


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(config)
        head_channels = 512
        if config.vit_post == 'mamba':
            self.mamba_block = nn.ModuleList()
            for i in range(config.vit_post_numlayers):
                self.mamba_block.append(mamba.VMAMBA2Block(dim=config.hidden_size,
                                                           input_resolution=(
                                                               14, 14),
                                                           num_heads=8,
                                                           linear_attn_duality=False,
                                                           ssd_chunk_size=config.ssd_chunk_size))
            # if config.vit_post_numlayers == 2:
            #     self.mamba_block2 = mamba.VMAMBA2Block(dim=config.hidden_size,
            #                                            input_resolution=(7, 7),
            #                                            num_heads=8,
            #                                            linear_attn_duality=False,
            #                                            ssd_chunk_size=config.ssd_chunk_size)

        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv1 = Conv2dReLU(
            1536,
            512,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv2 = Conv2dReLU(
            1536,
            640,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv3 = Conv2dReLU(
            1280,
            1,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv4 = Conv2dReLU(
            1280,
            256,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        if config.fpn == 'rpl':
            # self.conv_f1 = RPLinearBlock(1024)
            # self.conv_f2 = RPLinearBlock(1024)
            self.conv_f3 = RPLinearBlock(640, config.rpl_ratio)
        if config.fpn == 'da':
            self.conv_f3 = DA.DA_Block(256)
        if config.fpn == 'da_v2':
            self.conv_f3 = DA.DA_Block(128)

        if config.fpn == 'cga':
            self.conv1 = Conv2dReLU(
                1536,
                1024,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.conv2 = Conv2dReLU(
                1024,
                640,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.conv3 = Conv2dReLU(
                640,
                1,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.conv4 = Conv2dReLU(
                640,
                256,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.conv_1 = cga.DEABlockTrain(
                cga.default_conv,  dim=1024, kernel_size=3, reduction=16)
            self.conv_2 = cga.DEABlockTrain(
                cga.default_conv, dim=640, kernel_size=3, reduction=8)
            self.mix1 = cga.CGAFusion(dim=1024, reduction=16)
            self.mix2 = cga.CGAFusion(dim=640, reduction=8)

        elif config.fpn == 'msaa':
            self.conv1 = Conv2dReLU(
                1536,
                640,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.conv2 = Conv2dReLU(
                1024,
                640,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=8)
            self.fpn_method = MSAA.MSAA(in_channels=640, out_channels=1280)

        elif config.fpn == 'msaa_v2':
            # 使用DFBFormer的DSC2Block作为金字塔特征提取结构
            self.fpn_method = MSAA.MSAA(in_channels=640, out_channels=1280)
            self.conv1 = Conv2dReLU(
                1536,
                640,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.conv2 = Conv2dReLU(
                1024,
                640,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.conv_se2 = DSCB.DSC2Block(
                640,
                640,
                device=config.device,
                kernel_size=5
            )
            if config.upsample == 'dysample':
                self.se_up = DySample.DySample(640)
            else:
                self.se_up = nn.UpsamplingBilinear2d(scale_factor=2)

        elif config.fpn == 'msaa_v3':
            # DSC2Block改进为占用更少显存的DSC3Block
            self.fpn_method = MSAA.MSAA(in_channels=640, out_channels=1280)
            self.conv1 = Conv2dReLU(
                1536,
                640,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.conv2 = Conv2dReLU(
                1024,
                640,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
            self.conv_se2 = DSCB.DSC3Block(
                640,
                640,
                device=config.device,
                kernel_size=5
            )
            if config.upsample == 'dysample':
                self.se_up = DySample.DySample(640)
            else:
                self.se_up = nn.UpsamplingBilinear2d(scale_factor=2)

        if config.upsample == 'dysample':
            self.conv2_up = DySample.DySample(640)
            self.conv4_up = DySample.DySample(256)

    def forward(self, hidden_states, features=None):
        # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))

        # print(B, n_patch, hidden)  # 10, 196, 768

        # shortcut = hidden_states # 这个在之前跑的实验里没有，可以考虑加上
        if self.config.vit_post == 'mamba':

            # x输入B, N, hidden
            for mamba in self.mamba_block:
                hidden_states = mamba(hidden_states, H=h, W=w)
            # if self.config.vit_post_numlayers == 2:
            #     hidden_states = self.mamba_block2(
            #         hidden_states, H=h, W=w)
        # 这个mamba理论上应该同时使用双向的，先检查下mamba_block的实现逻辑，看看向量能不能在N上反向，然后再通过一次
        # hidden_states += shortcut
        # 修改部分，添加mamba_block

        x = hidden_states.permute(0, 2, 1)  # B, hidden, n_patch

        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)  # 这里是原来卷积模块在的位置 channel 512
        # if self.config.fpn == 'rpl':
        #     f1 = self.conv_f1(features[1])
        #     x = torch.cat([x, f1], dim=1)  # channel 1536
        # else:
        #     x = torch.cat([x, features[1]], dim=1)  # channel 1536
        x = torch.cat([x, features[1]], dim=1)
        # X is the output of ViT branch
        # features channel 1536, 1024, 640
        if self.config.fpn == 'cga':
            x = self.conv1(x)  # channel 1024
            x = self.conv_1(x)  # channel 1024
            x = self.up4(x)
            x = self.mix1(x, features[2])

            x = self.conv2(x)  # channel 640
            x = self.conv_2(x)
            x = self.up2(x)
            x = self.mix2(x, features[3])  # channel 1280

            x1 = self.conv3(x)  # channel 1
            x2 = self.conv4(x)  # channel 256

            x1 = self.up2(x1)
            x2 = self.up2(x2)
            x1 = torch.cat([x1, features[0]], dim=1)

            return x1, x2

        elif self.config.fpn == 'msaa':
            x_semantic = self.conv1(x)  # channel 640 14x14
            x_semantic = self.up4(x_semantic)
            x_middle = self.conv2(features[2])  # channel 640 56x56
            if self.config.upsample == 'dysample':
                x_middle = self.conv2_up(x_middle)
            else:
                x_middle = self.up2(x_middle)
            x_edge = features[3]  # channel 640 112x112

            x = self.fpn_method(x_semantic, x_middle, x_edge)
        elif self.config.fpn == 'msaa_v2' or self.config.fpn == 'msaa_v3':
            x_semantic = self.conv1(x)  # channel 640
            x_semantic = self.up4(x_semantic)
            x_semantic = self.conv_se2(x_semantic)
            x_semantic = self.se_up(x_semantic)

            x_middle = self.conv2(features[2])  # channel 640
            if self.config.upsample == 'dysample':
                x_middle = self.conv2_up(x_middle)
            else:
                x_middle = self.up2(x_middle)

            x_edge = features[3]  # channel 640
            # print(x_semantic.shape, x_middle.shape, x_edge.shape)
            x = self.fpn_method(x_semantic, x_middle, x_edge)
        else:
            x = self.conv1(x)  # channel 512
            x = self.up4(x)
            # if self.config.fpn == 'rpl':
            #     f2 = self.conv_f2(features[2])
            #     x = torch.cat([x, f2], dim=1)
            # else:
            #     x = torch.cat([x, features[2]], dim=1)  # channel 1536
            x = torch.cat([x, features[2]], dim=1)  # channel 1536
            x = self.conv2(x)  # channel 640
            if self.config.upsample == 'dysample':
                x = self.conv2_up(x)
            else:
                x = self.up2(x)
            if self.config.fpn == 'rpl':
                f3 = self.conv_f3(features[3])
                x = torch.cat([x, f3], dim=1)

            else:
                x = torch.cat([x, features[3]], dim=1)  # channel 1280

        x1 = self.conv3(x)  # channel 1
        x2 = self.conv4(x)  # channel 256
        if self.config.fpn == 'da':
            tmp = self.conv_f3(x2)
            x2 = x2+tmp
        if self.config.fpn == 'da_v2':
            tmp1, tmp2 = x2.chunk(2, dim=1)
            tmp2 = self.conv_f3(tmp2)
            x2 = torch.cat([tmp1, tmp2], dim=1)

        if self.config.upsample == 'dysample':
            x1 = self.up2(x1)
            x2 = self.conv4_up(x2)
        else:
            x1 = self.up2(x1)
            x2 = self.up2(x2)
        # Concat the 1-channal output_map of ViT branch to assist CNN branch training
        x1 = torch.cat([x1, features[0]], dim=1)

        return x1, x2


class VisionTransformer(nn.Module):
    def __init__(self, config, backbone, img_size=224, num_classes=17, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.width = backbone.width
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis, backbone)
        self.decoder = DecoderCup(config)

        self.segmentation_head1 = SegmentationHead(   # Classifier for CNN branch
            config,
            in_channels=self.width+1,
            out_channels=num_classes,
            kernel_size=3,
        )
        self.segmentation_head2 = SegmentationHead(   # Classifier for ViT branch
            config,
            in_channels=256,
            out_channels=num_classes,
            kernel_size=3,
        )
        if config.fpn == 'da_head':
            self.segmentation_head2 = DASegmentationHead(
                config,
                256,
                num_classes,
            )

        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x1, attn_weights, features = self.transformer(
            x)  # (B, n_patch, hidden)
        x1, x2 = self.decoder(x1, features)  # x1,
        logits1 = self.segmentation_head1(x1)
        logits2 = self.segmentation_head2(x2)
        return logits1, logits2

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" %
                            (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' %
                      (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(
                    np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'testing': configs.get_testing(),
}
