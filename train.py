import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_dataset
from networks.vit_seg_modeling_L2HNet import L2HNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='Chesapeake', help='experiment_name')
parser.add_argument('--max_epochs', type=int, default=100,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64,
                    help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
parser.add_argument('--save_path', type=str)
parser.add_argument('--gpu', type=str, help='Select GPU number to train')
parser.add_argument('--vit_post', type=str,
                    default='mamba', help='vit post process')
parser.add_argument('--vit_post_numlayers', type=int,
                    default=1, help='number of layers in vit_post')
parser.add_argument('--vit_numlayers', type=int,
                    default=12, help='number of layers in vit_post')
parser.add_argument('--upsample', type=str, default='original',
                    help='upsampling method')
parser.add_argument('--seghead', type=str, default=None,
                    help='segmentation head')
parser.add_argument('--fpn', type=str, default=None, help='fpn name')
parser.add_argument('--ssd_chunk_size', type=int, default=14,
                    help='whether use pretrained weight or not')
parser.add_argument('--preprocess', type=str,
                    default='DICAM', help='preprocess name')
parser.add_argument('--rpblock', type=str, default=None, help='L2HNet RPBlock')
parser.add_argument('--rpl_ratio', type=list,
                    default=[0.25, 0.5, 0.25], help='RPL ratio')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if __name__ == "__main__":
    vit_patches_size = 16
    img_size = 224
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Chesapeake': {  # default dataset as a example
            # The path of the *.csv file
            'list_dir': '/lustre/chaixiujuan/ChaiXin/Paraformer-main/dataset/CSV_list/Chesapeake_NewYork.csv',
            'num_classes': 17
        }
    }  # Create a config to your own dataset here

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    snapshot_path = args.save_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.n_classes = args.num_classes
    config_vit.vit_post = args.vit_post
    config_vit.vit_post_numlayers = args.vit_post_numlayers
    config_vit.transformer.num_layers = args.vit_numlayers
    config_vit.fpn = args.fpn
    config_vit.rpl_ratio = args.rpl_ratio
    config_vit.upsample = args.upsample
    config_vit.seghead = args.seghead
    config_vit.preprocess = args.preprocess
    config_vit.ssd_chunk_size = args.ssd_chunk_size
    config_vit.device = torch.device('cuda')
    config_vit.patches.grid = (
        int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, backbone=L2HNet(width=args.CNN_width, rpblock=args.rpblock),
                  img_size=img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    trainer_dataset(args, net, snapshot_path)
