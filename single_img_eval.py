import numpy as np
import torch
import rasterio


LABEL_CLASSES = [0, 11, 12, 21, 22, 23, 24,
                 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]


LABEL_2_BASE = {
    0: 0,
    11: 1,
    12: 1,
    21: 2,
    22: 2,
    23: 2,
    24: 2,
    31: 3,
    41: 4,
    42: 4,
    43: 4,
    52: 3,
    71: 3,
    81: 3,
    82: 3,
    90: 4,
    95: 3}
# 这里有蹊跷，为什么没有51号dwarf scrub？

GT_2_BASE = {
    0: 0,
    1: 1,
    2: 4,
    3: 3,
    4: 2,
    5: 2,
    6: 2,
    15: 0}

label_names = {0: 'background', 1: 'water',
               2: 'tree', 3: 'low vegetation', 4: 'built-up'}

IDX_TO_LABEL = {
    idx: c for idx, c in enumerate(LABEL_CLASSES)
}


TRANS_TYPE = {'high_quality': [GT_2_BASE], 'low_quality': [
    LABEL_2_BASE], 'prediction': [IDX_TO_LABEL, LABEL_2_BASE]}


def data_trans(data, trans_type='high_quality'):
    for trans in TRANS_TYPE[trans_type]:
        tmp = np.zeros_like(data)
        for gt_label, base_label in trans.items():
            tmp[data == gt_label] = base_label
        data = tmp

    value_counts = np.unique(data, return_counts=True)
    # print('trans_type: {}, value_counts: {}'.format(trans_type, value_counts))

    return data


def _fast_hist(label_true, label_pred, n_class):
    """生成混淆矩阵

    Args:
        label_true (_type_): _description_
        label_pred (_type_): _description_
        n_class (_type_): 需要计算背景类，所以n_class+1

    Returns:
        _type_: _description_
    """

    # label_true = label_true.numpy()
    # label_pred = label_pred.numpy()

    # mask在元素值大于等于0，小于等于n_class的地方填True
    mask = (label_true >= 0) & (label_true < n_class)
    # print('mask:\n', mask)

    # label_true[mask]用mask中为true的索引取label_true
    # print('label_true[mask]:\n', label_true[mask].astype(int))

    # 以下个人理解为相当于编码过程,label_true看成'十位',label_pred看成'个位'
    # 将5*5的矩阵与0-24的地址对应上
    # print('n_class * label_true[mask]:\n',
    #       n_class * label_true[mask].astype(int))
    # print('label_pred[mask]:\n', label_pred[mask])

    # print('n_class * label_true[mask].astype(int) + label_pred[mask]:\n',
    #       n_class * label_true[mask].astype(int) + label_pred[mask])

    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)

    # print('hist:\n', hist)

    return hist


def calculate_iou_and_acc(pred, target, num_classes, cal_flag=True):
    """计算IoU和acc

    Args:
        pred (_type_): 单个预测图像
        target (_type_): _description_
        num_classes (_type_): _description_
        cal_flag (bool, optional): True表示计算IoU和acc，False表示不计算，False时返回hist，True返回hist和计算结果的字典. Defaults to True.

    Returns:
        _type_: _description_
    """

    hist_num_classes = num_classes + 1

    # hist = np.zeros((hist_num_classes, hist_num_classes))

    # 一个batch里可能有多个数据，通过迭代器逐个计算
    # for p, t in zip(pred, target):
    #     hist += _fast_hist(p.flatten(), t.flatten(), hist_num_classes)
    hist = _fast_hist(target.flatten(), pred.flatten(), hist_num_classes)
    # 其实这里也可以不flatten, 函数值arr[mask]这一步也可帮忙拉平

    # print('hist:\n', hist)

    if not cal_flag:
        return hist

    acc = np.diag(hist).sum() / hist.sum()  # 混淆矩阵对角线元素和 / 所有元素和
    # hist.sum(axis=1) => [3,3,1,1,1],对应每一类真实标签数量
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)  # hist.sum(axis=0) => [2,3,1,2,1],对应每一类预测标签数量

    # freq = hist.sum(axis=1) / hist.sum()
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()  # 没弄懂是什么意思

    output = {
        'mIoU': mean_iu,
        'acc': acc,
        'iou': iu,
        'acc_cls': acc_cls,
        'hist': hist
    }

    return output


def load_rasterio_img(img_path):
    image = rasterio.open(img_path, "r")
    img = image.read().squeeze()
    img_data = {
        'img': img,
        'meta': image.meta,
        'crs': image.crs.wkt,
        'shape': image.shape}

    return img_data


def evaluate_single_img(pre, target, num_classes, cal_flag=True):
    """
    pre: 预测结果
    target: 人工标注结果
    num_classes: 类别
    先统一pre和target的尺寸，然后计算IoU和准确率
    """
    # target = torch.from_numpy(target)
    # pre = torch.from_numpy(pre)

    if target.shape != pre.shape:
        print("The shapes of the arrays are different.")
        if pre.shape[0] > target.shape[0]:
            large_arr = pre
            small_arr = target
            # Calculate the center coordinates of the large array
            center_x = large_arr.shape[0] // 2
            center_y = large_arr.shape[1] // 2

            # Calculate the top-left coordinates of the region to extract
            x_start = center_x - small_arr.shape[0] // 2
            y_start = center_y - small_arr.shape[1] // 2

            # Extract the region from the large array
            pre = large_arr[x_start:x_start +
                            small_arr.shape[0], y_start:y_start+small_arr.shape[1]]
        else:
            large_arr = target
            small_arr = pre
            # Calculate the center coordinates of the large array
            center_x = large_arr.shape[0] // 2
            center_y = large_arr.shape[1] // 2

            # Calculate the top-left coordinates of the region to extract
            x_start = center_x - small_arr.shape[0] // 2
            y_start = center_y - small_arr.shape[1] // 2

            # Extract the region from the large array
            target = large_arr[x_start:x_start+small_arr.shape[0],
                               y_start:y_start+small_arr.shape[1]]

    return calculate_iou_and_acc(pre, target, num_classes, cal_flag=cal_flag)


if __name__ == '__main__':
    import os

    label_pred = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'm_4307417_sw_18_1_predictions-new.tif')

    high_quality_label = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'm_4307417_sw_18_1_lc.tif')
    low_quality_label = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'm_4307417_sw_18_1_nlcd.tif')
    img_new = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'm_4307417_sw_18_1_naip-new.tif')

    high_quality_label_img = load_rasterio_img(high_quality_label)
    low_quality_pred_img = load_rasterio_img(label_pred)
    image_new = load_rasterio_img(img_new)
    low_quality_label_img = load_rasterio_img(low_quality_label)

    high_quality_label_img = high_quality_label_img["img"]
    low_quality_pred_img = low_quality_pred_img["img"]
    low_quality_label_img = low_quality_label_img["img"]

    print("*******************")

    base_GT = data_trans(high_quality_label_img, trans_type='high_quality')

    base_low = data_trans(low_quality_label_img, trans_type='low_quality')

    base_pre = data_trans(low_quality_pred_img, trans_type='prediction')

    # 示例使用
    base_classes = 4
    eva_result = evaluate_single_img(base_pre, base_GT, base_classes)
    # print(eva_result)
    print('the prediction evaluation result is :')
    print('mIou is {}, mAcc is {}, iou is {}'.format(
        eva_result['mIoU'], eva_result['acc'], eva_result['iou']))

    low_quality_label_img_eva_result = evaluate_single_img(
        base_low, base_GT, base_classes)
    # print(eva_result)
    print('the low quality img evaluation result is :')
    print('mIou is {}, mAcc is {}, iou is {}'.format(
        low_quality_label_img_eva_result['mIoU'],
        low_quality_label_img_eva_result['acc'],
        low_quality_label_img_eva_result['iou']))
