import os
import sys
import logging
import numpy as np
import argparse
from single_img_eval import evaluate_single_img, load_rasterio_img, data_trans

NUM_CLASSES = 4


def calculate_metrics_from_hist(hist):
    # 混淆矩阵第一行和第一列均为0，需要先去掉，不去掉好像不影响结果
    # hist = hist[1:, 1:]
    print("**************")
    print('Final metrics hist:')
    print(hist)
    print("**************")
    acc = np.diag(hist).sum() / hist.sum()  # 混淆矩阵对角线元素和 / 所有元素和
    # hist.sum(axis=1) => [3,3,1,1,1],对应每一类真实标签数量
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)  # hist.sum(axis=0) => [2,3,1,2,1],对应每一类预测标签数量
    output = {
        'mIoU': mean_iu,
        'acc': acc,
        'iou': iu,
        'acc_cls': acc_cls
    }
    return output


def main(target_dir=None,
         target_trans_type='high_quality',
         result_dir=None,
         result_trans_type='prediction',
         num_classes=NUM_CLASSES,
         cal_flag=False):
    hist_num_classes = num_classes+1
    hist = np.zeros((hist_num_classes, hist_num_classes))

    for gt_file_name, pred_file_name in zip(target_dir, result_dir):
        if file_name.endswith('.tif'):
            gt_img = load_rasterio_img(gt_file_name)["img"]
            gt_map = data_trans(gt_img, target_trans_type)

            pre_img = load_rasterio_img(pred_file_name)["img"]
            pre_map = data_trans(pre_img, result_trans_type)

            hist += evaluate_single_img(
                gt_map,
                pre_map,
                num_classes,
                cal_flag=cal_flag)

    return calculate_metrics_from_hist(hist)


parser = argparse.ArgumentParser()

parser.add_argument('--result_dir', type=str,
                    default='../test_result/baseline')  # 预测文件目录
parser.add_argument('--result_suffix', type=str, default='_predictions-new')
parser.add_argument('--result_trans_type', type=str,
                    default='prediction')  # 预测文件转换类型
parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)  # 分类数目
parser.add_argument('--cal_flag', type=bool, default=False)  # 是否在单个图像上计算混淆矩阵
parser.add_argument('--target_dir', type=str,
                    default='./dataset/Chesapeake_NewYork_dataset/HR_label'),
parser.add_argument('--target_trans_type', type=str,
                    default='high_quality')
parser.add_argument('--target_suffix', type=str, default='_lc')
args = parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(filename=os.path.join(args.result_dir, "eval_log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    hr_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), args.target_dir)
    lr_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), args.result_dir)
    hr_file_list = os.listdir(hr_dir)
    tif_file_list = [
        file_name for file_name in hr_file_list if file_name.endswith('.tif')]
    hr_tif_list = []
    lr_tif_list = []
    for file_name in tif_file_list:
        hr_tif_list.append(os.path.join(hr_dir, file_name))
        lr_tif_list.append(os.path.join(
            lr_dir, file_name.replace(args.target_suffix, args.result_suffix)))

    results = main(target_dir=hr_tif_list,
                   target_trans_type=args.target_trans_type,
                   result_dir=lr_tif_list,
                   result_trans_type=args.result_trans_type,
                   num_classes=args.num_classes,
                   cal_flag=args.cal_flag
                   )
    logging.info('evaluate result:{}'.format(results))
    print('mIoU:{:.4f}, mAcc:{:.4f}'.format(results['mIoU'], results['acc_cls']))
