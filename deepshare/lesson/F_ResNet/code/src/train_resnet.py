# -*- coding: utf-8 -*-
"""
# @file name  : train_resnet.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-06-23
# @brief      : resnet training on cifar10
"""
from PIL import Image
import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from F_ResNet.code.tools.cifar10_dataset import CifarDataset
from F_ResNet.code.tools.resnet import resnet56, resnet20
from F_ResNet.code.tools.common_tools import ModelTrainer, show_confMat, plot_line
from matplotlib import pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train):
        img_ = np.array(img_) * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_

if __name__ == "__main__":

    # config
    # train_dir = os.path.join(BASE_DIR, "..", "..","..", "Data", "cifar-10",  "cifar10_train")
    # test_dir = os.path.join(BASE_DIR, "..", "..","..", "Data", "cifar-10", "cifar10_test")
    train_dir = os.path.join(BASE_DIR, "..", "..", "..", "Data", "cifar-10", "traindata")
    test_dir = os.path.join(BASE_DIR, "..", "..", "..", "Data", "cifar-10", "testdata")
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_names=('cassiterite','nepheline','Spinel')
    #num_classes = 10
    num_classes=3
    MAX_EPOCH=50
    #MAX_EPOCH = 182     # 182     # 64000 / (45000 / 128) = 182 epochs
    #总iteration数64000 * batchsize128 =epoch *总datasize（训练集）=总训练了多少条数据（包括一条数据多次训练）
    BATCH_SIZE = 128
    LR = 0.1
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    milestones = [92, 136]  # divide it by 10 at 32k and 48k iterations  在这两个点降学习率

    # ============================ step 1/5 数据 ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建MyDataset实例
    train_data = CifarDataset(data_dir=train_dir, transform=train_transform)
    valid_data = CifarDataset(data_dir=test_dir, transform=valid_transform)
    print(train_data.__len__())#########

    ##瞎测
    img_tensor,label=train_data.__getitem__(66)#class CifarDataset(Dataset):
    img_rgb = transform_invert(img_tensor, train_transform)
    print(img_tensor, label)
    print(img_rgb)
    plt.imshow(img_rgb)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=2)

    # ============================ step 2/5 模型 ============================
    resnet_model = resnet20()

    resnet_model.to(device)
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    optimizer = optim.SGD(resnet_model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)  # 选择优化器

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)

# ============================ step 5/5 训练 ============================
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, resnet_model, criterion, optimizer, epoch, device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, resnet_model, criterion, device)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
            epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        scheduler.step()  # 更新学习率

        # 绘图
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == MAX_EPOCH-1)
        show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == MAX_EPOCH-1)#画混淆阵

        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)#画loss曲线acc曲线

        if epoch > (MAX_EPOCH/2) and best_acc < acc_valid:#精度比较高就保存学习砍pytorch模型保存加载
            best_acc = acc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": resnet_model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                      best_acc, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)






