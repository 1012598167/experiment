# -*- coding: utf-8 -*-
"""
# @file name  : resnet-inference.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : inference demo
"""

import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
import time
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from F_ResNet.code.tools.common_tools import get_resnet_18, get_resnet_50
BASE_DIR = os.path.dirname(os.path.abspath(__file__))###路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")##GPU
print(device)
print(BASE_DIR)

def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def process_img(path_img):#图像预处理
    #将图片处理到与训练时格式一样
    # hard code
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([#变换的组成
        transforms.Resize(256),#大小
        transforms.CenterCrop((224, 224)),#分辨率设置一样
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')
    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)
    # tensor([[[[1.6667, 1.7352, 1.8379, ..., 2.0605, 2.0434, 2.1119],
    #           [1.7694, 1.7865, 1.7865, ..., 2.0605, 2.1119, 2.0092],
    #           [1.5468, 1.5810, 1.6324, ..., 2.0092, 2.0092, 1.9749], ...,
    #           [0.6906, 0.5022, 0.6049, ..., 1.1187, 1.4783, 1.3242],
    #           [0.6049, 0.8104, 0.8618, ..., 1.1187, 1.4269, 1.3413],
    #           [0.9988, 1.2557, 1.0844, ..., 1.2043, 1.4612, 1.3242]],
    #          [[0.9405, 1.0105, 1.1155, ..., 1.2731, 1.2556, 1.3081],
    #           [1.0630, 1.0805, 1.0805, ..., 1.3256, 1.3256, 1.2031],
    #           [0.9230, 0.9580, 1.0105, ..., 1.2731, 1.2731, 1.1856], ...,
    #           [0.6954, 0.5028, 0.6078, ..., 0.2402, 0.6429, 0.5903],
    #           [0.6078, 0.8179, 0.8704, ..., 0.2402, 0.5903, 0.6254],
    #           [0.9755, 1.2381, 1.0630, ..., 0.3277, 0.6254, 0.6078]],
    #          [[0.2348, 0.3045, 0.4091, ..., 0.5136, 0.4962, 0.5311],
    #           [0.3568, 0.3742, 0.3742, ..., 0.5311, 0.5485, 0.4265],
    #           [0.1825, 0.2173, 0.2696, ..., 0.4788, 0.4788, 0.4091], ...,
    #           [-0.1835, -0.3753, -0.2707, ..., -0.7238, -0.3055, -0.3404],
    #           [-0.2707, -0.0615, -0.0092, ..., -0.7238, -0.3578, -0.3578],
    #           [0.1302, 0.3916, 0.2173, ..., -0.6018, -0.3230, -0.3753]]]], device='cuda:0')
    #torch.Size([1, 3, 224, 224]#batchsize 通道数 高 宽

    img_tensor.unsqueeze_(0)        # chw --> bchw
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding='UTF-8') as f:  # 设置文件对象
        class_names_cn = f.readlines()
    return class_names, class_names_cn


if __name__ == "__main__":

    # config
    path_state_dict_18 = os.path.join(BASE_DIR, "..", "data", "resnet18-5c106cde.pth")
    path_state_dict_50 = os.path.join(BASE_DIR, "..", "data", "resnet50-19c8e357.pth")
    #path_state_dict_50 = os.path.join(BASE_DIR, "..", "data", "checkpoint_best.pkl")
    path_img = os.path.join(BASE_DIR, "..", "data","Golden Retriever from baidu.jpg")
     #path_img = os.path.join(BASE_DIR, "..", "data", "tiger cat.jpg")
   # path_img = os.path.join(BASE_DIR, "..", "data", "cassiterite.jpg")
    path_classnames = os.path.join(BASE_DIR, "..", "data", "imagenet1000.json")
    #path_classnames = os.path.join(BASE_DIR, "..", "data", "mineral3.json")
    path_classnames_cn = os.path.join(BASE_DIR, "..", "data", "imagenet_classnames.txt")
    #path_classnames_cn = os.path.join(BASE_DIR, "..", "data", "mineral3_classnames.txt")
    # load class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 1/5 load img
    img_tensor, img_rgb = process_img(path_img)

    # 2/5 load model
    #resnet_model = get_resnet_18(path_state_dict_18, device, True)

    resnet_model = get_resnet_50(path_state_dict_50, device, True)#可以试着print出来 和论文里resnet 50-layer的列数据一致
    print("########resnet_model:#######")
    print(resnet_model)
    print("############################")

    # ResNet(
    #   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#conv1 #输入3卷积核输出64卷积核 stride2*2 第一层标配
    #   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)#affine参数设为True表示weight和bias将被使用 这一步就是将参数标准化
    #   (relu): ReLU(inplace=True)DataParallel
    #   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)#注意stride=2
    #   (layer1): Sequential(  #conv2.x
    #     (0): BasicBlock(
    #       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     )
    #     (1): BasicBlock(
    #       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     )
    #   )#【3*3,64  3*3,64】*2
    #   (layer2): Sequential(#conv3.x
    #     (0): BasicBlock(
    #       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)#stride变了 改变了特征图分辨率
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (downsample): Sequential(#所以+x的时候维数不同了 要downsample
    #         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    #         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       )
    #     )
    #     (1): BasicBlock(
    #       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     )
    #   )
    #............................

    #一个basic block两个卷积层
    # 3/5 inference  tensor --> vector
    with torch.no_grad():
        time_tic = time.time()
        outputs = resnet_model(img_tensor)
        time_toc = time.time()

    # 4/5 index to class names
    _, pred_int = torch.max(outputs.data, 1)
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)

    pred_idx = int(pred_int.cpu().numpy())
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]
    print("img: {} is: {}\n{}".format(os.path.basename(path_img), pred_str, pred_cn))
    #img: Golden Retriever from baidu.jpg is: golden retriever
    #207 n02099601 狗, golden retriever
    print("time consuming:{:.2f}s".format(time_toc - time_tic))
    #time consuming:1.79s

    # 5/5 visualization
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    top5_num = top5_idx.cpu().numpy().squeeze()
    text_str = [cls_n[t] for t in top5_num]
    for idx in range(len(top5_num)):
        plt.text(5, 15+idx*30, "top {}:{}".format(idx+1, text_str[idx]), bbox=dict(fc='yellow'))
    plt.show()#图片输出出来看一看

