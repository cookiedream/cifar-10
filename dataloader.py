import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data

import os

if not os.path.exists('./cifar10'):
    os.mkdir('./cifar10')

# 設置超參數
DOANLOAD_DATASET = False
BATCH_SIZE = 256

# 預先處理資料
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 載入資料
train_data = torchvision.datasets.CIFAR10(
    root='./cifar10',
    train=True,
    transform=train_transform,
    download=DOANLOAD_DATASET
)

test_data = torchvision.datasets.CIFAR10(
    root='./cifar10',
    train=False,
    transform=test_transform,
    download=DOANLOAD_DATASET
)

# 把要訓練的資料放入DataLoader
data_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,  # 建議在測試時也使用一個較小的batch_size，而不是一次載入全部數據
    shuffle=False
)
test_x, test_y = next(iter(test_loader))

# cifar10的類別
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
