import argparse
import rasterio
from rasterio.errors import RasterioIOError
from torch.utils.data.dataset import Dataset
import os
import random
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch.nn.functional as F
import utils
import torch
from networks.vit_seg_modeling_L2HNet import L2HNet

from train_utils import evaluate as batch_evaluate


class TileEvaluateDatasetate(Dataset):

    def __init__(self, fn, chip_size, stride, transform=None, windowed_sampling=False, verbose=False):
        self.fn = fn
        self.chip_size = chip_size

        self.transform = transform
        self.windowed_sampling = windowed_sampling
        self.verbose = verbose

        with rasterio.open(self.fn) as f:
            height, width = f.height, f.width
            self.num_channels = f.count
            self.dtype = f.profile["dtype"]
            if not windowed_sampling:  # if we aren't using windowed sampling, then go ahead and read in all of the data
                self.data = np.rollaxis(f.read(), 0, 3)

        # upper left coordinate (y,x), of each chip that this Dataset will return
        self.chip_coordinates = []
        for y in list(range(0, height - self.chip_size, stride)) + [height - self.chip_size]:
            for x in list(range(0, width - self.chip_size, stride)) + [width - self.chip_size]:
                self.chip_coordinates.append((y, x))
        self.num_chips = len(self.chip_coordinates)

        if self.verbose:
            print("Constructed TileEvaluateDatasetate -- we have %d by %d file with %d channels with a dtype of %s. We are sampling %d chips from it." % (
                height, width, self.num_channels, self.dtype, self.num_chips
            ))

    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]

        if self.windowed_sampling:
            try:
                with rasterio.Env():
                    with rasterio.open(self.fn) as f:
                        img = np.rollaxis(f.read(window=rasterio.windows.Window(
                            x, y, self.chip_size, self.chip_size)), 0, 3)
            # NOTE(caleb): I put this here to catch weird errors that I was seeing occasionally when trying to read from COGS - I don't remember the details though
            except RasterioIOError as e:
                print("Reading %d failed, returning 0's" % (idx))
                img = np.zeros((self.chip_size, self.chip_size,
                               self.num_channels), dtype=np.uint8)
        else:
            img = self.data[y:y+self.chip_size, x:x+self.chip_size]

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array((y, x))

    def __len__(self):
        return self.num_chips


parser = argparse.ArgumentParser()
CHIP_SIZE = 224
PADDING = 112
assert PADDING % 2 == 0
HALF_PADDING = PADDING//2
CHIP_STRIDE = CHIP_SIZE - PADDING
parser.add_argument('--dataset', type=str,
                    default='Chesapeake', help='experiment_name')
parser.add_argument('--max_epochs', type=int, default=30,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224,
                    help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64,
                    help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
parser.add_argument('--save_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--gpu', type=str, help='Select GPU number to train')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def get_label_fns(args, image_fns):
    label_fns = []
    for image_fn in image_fns:
        label_fn = image_fn.replace(".tif", "_labels.tif")
        assert os.path.exists(label_fn)
        label_fns.append(label_fn)
    return label_fns


def evaluate(args, model, test_save_path=None):
    model.eval()
    input_dataframe = pd.read_csv(args.list_dir)
    image_fns = input_dataframe["image_fn"].values
    label_fns = get_label_fns(args, image_fns)

    for image_idx in range(len(image_fns)):
        image_fn = image_fns[image_idx]
        label_fn = label_fns[image_idx]

        print("(%d/%d) Processing %s" %
              (image_idx, len(image_fns), image_fn), end=" ... ")
        print("(%d/%d) Processing %s" %
              (image_idx, len(image_fns), label_fn), end=" ... ")
        # -------------------
        # Load input and create dataloader
        # -------------------

        def image_transforms(img):
            img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
            img = np.rollaxis(img, 2, 0).astype(np.float32)
            img = torch.from_numpy(img)
            return img

        with rasterio.open(image_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        val_img_dataset = TileEvaluateDatasetate(
            image_fn,
            chip_size=CHIP_SIZE,
            stride=CHIP_STRIDE,
            transform=image_transforms,
            verbose=False)
        val_lab_dataset = TileEvaluateDatasetate(
            label_fn,
            chip_size=CHIP_SIZE,
            stride=CHIP_STRIDE,
            transform=None,
            verbose=False)

        # 这一行是干啥的？
        dataset = torch.utils.data.ConcatDataset(
            [val_img_dataset, val_lab_dataset])

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=4,
            pin_memory=True,
        )

        confmat = batch_evaluate(model, dataloader,
                                 device="cuda",
                                 num_classes=args.num_classes+1)
        val_info = str(confmat)
        print(val_info)

        # -------------------
        # Run model and organize output
        # -------------------

        # output = np.zeros((args.num_classes, input_height,
        #                   input_width), dtype=np.float32)
        # kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        # kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        # counts = np.zeros((input_height, input_width), dtype=np.float32)

        # for i, (data, coords) in enumerate(dataloader):
        #     data = data.cuda()
        #     with torch.no_grad():
        #         t_output1, t_output2 = model(data)
        #         # Fuse two branches outputs
        #         t_output = F.softmax(
        #             ((t_output1+t_output2)/2), dim=1).cpu().numpy()

        #     for j in range(t_output.shape[0]):
        #         y, x = coords[j]

        #         output[:, y:y+CHIP_SIZE, x:x+CHIP_SIZE] += t_output[j] * kernel
        #         counts[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += kernel

        # output = output / counts
        # output_hard = output.argmax(axis=0).astype(np.uint8)

        # -------------------
        # Save output
        # -------------------
        # output_profile = input_profile.copy()
        # output_profile["driver"] = "GTiff"
        # output_profile["dtype"] = "uint8"
        # output_profile["count"] = 1
        # output_profile["nodata"] = 0

        # output_fn = image_fn.split("/")[-1]
        # output_fn = output_fn.replace(
        #     "naip", "predictions")  # name the predictions
        # output_fn = os.path.join(test_save_path, output_fn)

        # with rasterio.open(output_fn, "w", **output_profile) as f:
        #     f.write(output_hard, 1)
        #     f.write_colormap(1, utils.LABEL_IDX_COLORMAP)
    return "Testing Finished!"


if __name__ == "__main__":
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Chesapeake': {
            # The path of the *.csv file
            'list_dir': '/lustre/chaixiujuan/ChaiXin/Paraformer-main/dataset/CSV_list/Chesapeake_NewYork.csv',
            'num_classes': 17
        }
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    vit_patches_size = 16
    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.n_classes = args.num_classes
    config_vit.patches.size = (vit_patches_size, vit_patches_size)
    config_vit.patches.grid = (
        int(args.img_size/vit_patches_size), int(args.img_size/vit_patches_size))
    net = ViT_seg(config_vit, backbone=L2HNet(width=args.CNN_width),
                  img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    snapshot = args.model_path
    net.load_state_dict(torch.load(snapshot))
    # test_save_path = args.save_path
    # os.makedirs(test_save_path, exist_ok=True)

    evaluate(args, net)


# import torch
# from train_utils import evaluate
# import transforms as T
# import trainer
# import pandas as pd


# class SegmentationPresetTrain:
#     def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#         min_size = int(0.5 * base_size)
#         max_size = int(2.0 * base_size)

#         trans = [T.RandomResize(min_size, max_size)]
#         if hflip_prob > 0:
#             trans.append(T.RandomHorizontalFlip(hflip_prob))
#         trans.extend([
#             T.RandomCrop(crop_size),
#             T.ToTensor(),
#             T.Normalize(mean=mean, std=std),
#         ])
#         self.transforms = T.Compose(trans)

#     def __call__(self, img, target):
#         return self.transforms(img, target)


# class SegmentationPresetEval:
#     def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#         self.transforms = T.Compose([
#             T.RandomResize(base_size, base_size),
#             T.ToTensor(),
#             T.Normalize(mean=mean, std=std),
#         ])

#     def __call__(self, img, target):
#         return self.transforms(img, target)


# def get_transform(train):
#     base_size = 520
#     crop_size = 480

#     return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


# def val_dataset_generator(root, split, mode, transforms=None):
#     # TODO: implement
#     return None


# def create_model(aux=False, num_classes=21):
#     return None


# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser(description="pytorch fcn training")

#     parser.add_argument("--data-path", default="/data/", help="data_root")
#     parser.add_argument("--weights", default="./save_weights/model_29.pth")
#     parser.add_argument("--list-dir", default="./data/train.csv")
#     parser.add_argument("--num-classes", default=20, type=int)
#     parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
#     # parser.add_argument("--device", default="cuda", help="training device")
#     # parser.add_argument('--print-freq', default=10,
#     #                     type=int, help='print frequency')

#     args = parser.parse_args()

#     return args


# if __name__ == '__main__':
#     args = parse_args()

#     num_workers = 4
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     num_classes = args.num_classes + 1

#     input_dataframe = pd.read_csv(args.list_dir)
#     image_fns = input_dataframe["image_fn"].values
#     label_fns = input_dataframe["label_fn"].values
#     NUM_CHIPS_PER_TILE = 50  # How many chips will be sampled from one large-scale tile
#     CHIP_SIZE = 224  # Size of each sampled chip
#     val_dataset = trainer.StreamingGeospatialDataset(
#         imagery_fns=image_fns,
#         label_fns=label_fns,
#         groups=None,
#         chip_size=CHIP_SIZE,
#         num_chips_per_tile=NUM_CHIPS_PER_TILE,
#         windowed_sampling=True,
#         verbose=False,
#         image_transform=trainer.image_transforms,
#         label_transform=trainer.label_transforms,
#         nodata_check=trainer.nodata_check
#     )
#     # val_dataset = val_dataset_generator(
#     #     root='data/VOCdevkit', split='val', mode='fine')

#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                              batch_size=1,
#                                              num_workers=num_workers,
#                                              pin_memory=True,
#                                              collate_fn=val_dataset.collate_fn)

#     model = create_model(aux=args.aux, num_classes=num_classes)
#     model.to(device)
#     confmat = evaluate(model, val_loader, device=device,
#                        num_classes=num_classes)
#     val_info = str(confmat)
#     print(val_info)
