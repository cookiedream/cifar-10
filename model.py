import torch.nn.functional as F
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN, self).__init__()
        self.num_classes = num_classes

        # in[N, 3, 32, 32] => out[N, 16, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
        # in[N, 16, 16, 16] => out[N, 32, 8, 8]
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2)

        )
        # in[N, 32 * 8 * 8] => out[N, 128]
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(True)
        )
        # in[N, 128] => out[N, 64]
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True)
        )
        # in[N, 64] => out[N, 10]
        self.out = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # [N, 32 * 8 * 8]
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)
        return output

# ----------------------------


class CNN3(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3, self).__init__()
        self.num_classes = num_classes

        # 第一层卷积：输入[N, 3, 32, 32]，输出[N, 16, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第二层卷积：输入[N, 16, 16, 16]，输出[N, 32, 8, 8]
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第三层卷积：输入[N, 32, 8, 8]，输出[N, 64, 4, 4]
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层1：输入[N, 64 * 4 * 4]，输出[N, 256]
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        # 全连接层2：输入[N, 64 * 4 * 4]，输出[N, 128]
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        # 全连接层3：输入[N, 128]，输出[N, 64]
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)
        # 输出层：输入[N, 64]，输出[N, num_classes]
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output


# ----------------------------
class CNN4(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN4, self).__init__()
        self.num_classes = num_classes

        # 第一层卷积：输入[N, 3, 32, 32]，输出[N, 16, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第二层卷积：输入[N, 16, 16, 16]，输出[N, 32, 8, 8]
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第三层卷积：输入[N, 32, 8, 8]，输出[N, 64, 4, 4]
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第四层卷积：输入[N, 64, 4, 4]，输出[N, 128, 4, 4]
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.relu_fc1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 256)
        self.relu_fc2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, 128)
        self.relu_fc3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(128, 64)
        self.relu_fc4 = nn.ReLU(inplace=True)
        # 输出层：输入[N, 64]，输出[N, num_classes]
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.fc4(x)
        x = self.relu_fc4(x)
        output = self.out(x)
        return output


# ----------------------------


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# ----------------------------


class CNN3_16(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_16, self).__init__()
        self.num_classes = num_classes

        # 第一层卷积：输入[N, 3, 32, 32]，输出[N, 16, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第二层卷积：输入[N, 16, 16, 16]，输出[N, 32, 8, 8]
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第三层卷积：输入[N, 32, 8, 8]，输出[N, 64, 4, 4]
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层1：输入[N, 64 * 4 * 4]，输出[N, 256]
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        # 全连接层2：输入[N, 64 * 4 * 4]，输出[N, 128]
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        # 全连接层3：输入[N, 128]，输出[N, 64]
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)
        # 输出层：输入[N, 64]，输出[N, num_classes]
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output


# ----------------------------


class CNN3_32(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_32, self).__init__()
        self.num_classes = num_classes

        # 第一層卷积：输入[N, 3, 32, 32]，输出[N, 32, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第二層卷积和第三層卷积的out_channels也更改為32
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层的输入尺寸调整为32 * 4 * 4
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output


# ----------------------------


class CNN3_64(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_64, self).__init__()
        self.num_classes = num_classes

        # 第一層卷积：输入[N, 3, 32, 32]，输出[N, 64, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第二層卷积和第三層卷积的out_channels也更改為64
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层的输入尺寸调整为64 * 4 * 4
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output


# ----------------------------

class CNN3_44(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_44, self).__init__()
        self.num_classes = num_classes

        # 第一层卷积：输入[N, 3, 32, 32]，输出[N, 16, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第二层卷积：输入[N, 16, 16, 16]，输出[N, 32, 8, 8]
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第三层卷积：输入[N, 32, 8, 8]，输出[N, 64, 4, 4]
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层1：输入[N, 64 * 4 * 4]，输出[N, 256]
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        # 全连接层2：输入[N, 64 * 4 * 4]，输出[N, 128]
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        # 全连接层3：输入[N, 128]，输出[N, 64]
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)
        # 输出层：输入[N, 64]，输出[N, num_classes]
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output

    # ----------------------------


class CNN3_77(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_77, self).__init__()
        self.num_classes = num_classes

        # 第一层卷积：输入[N, 3, 32, 32]，输出[N, 16, 16, 16]
        # 更改kernel_size为7, 调整padding以维持相同的输出尺寸
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第二层卷积：输入[N, 16, 16, 16]，输出[N, 32, 8, 8]
        # 更改kernel_size为7, 调整padding以维持相同的输出尺寸
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 第三层卷积：输入[N, 32, 8, 8]，输出[N, 64, 4, 4]
        # 更改kernel_size为7, 调整padding以维持相同的输出尺寸
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层1：输入[N, 64 * 4 * 4]，输出[N, 256]
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        # 全连接层2：输入[N, 256]，输出[N, 128]
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        # 全连接层3：输入[N, 128]，输出[N, 64]
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)
        # 输出层：输入[N, 64]，输出[N, num_classes]
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output


# ----------------------------


class CNN3_MaxPool3(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_MaxPool3, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)  # 修改这里
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)  # 修改这里
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)  # 修改这里
        )
        # 注意: 根据MaxPool2d的改变，全连接层的输入尺寸可能需要调整
        # 这里假设全连接层的输入尺寸不变，但实际上可能需要根据新的输出尺寸进行调整
        self.fc1 = nn.Linear(64, 256)  # 需要根据实际输出尺寸调整
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output

# ----------------------------


class CNN3_MaxPool4(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_MaxPool4, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)  # 保留这里的修改
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4)  # 保留这里的修改
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # 在这里不使用kernel_size=4的池化，避免输出尺寸过小
            nn.MaxPool2d(kernel_size=2, stride=2)  # 调整为较小的池化kernel_size
        )
        # 根据实际的输出尺寸调整全连接层的输入尺寸
        self.fc1 = nn.Linear(64, 256)  # 注意调整全连接层的输入尺寸
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU(inplace=True)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output


# ----------------------------


class CNN3_TwoHiddenLayers(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_TwoHiddenLayers, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(256, num_classes)  # 第二个全连接层直接输出类别数

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        output = self.fc2(x)
        return output


class CNN3_TwoHiddenLayers(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_TwoHiddenLayers, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(256, num_classes)  # 第二个全连接层直接输出类别数

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        output = self.fc2(x)
        return output

# ----------------------------


class CNN3_FourHiddenLayers(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_FourHiddenLayers, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # 扩大第一个全连接层的输出
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)  # 新增的第二个全连接层
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 128)  # 原先的第二个全连接层
        self.relu_fc3 = nn.ReLU()

        self.fc4 = nn.Linear(128, 64)  # 新增的第四个全连接层
        self.relu_fc4 = nn.ReLU()

        self.out = nn.Linear(64, num_classes)  # 输出层

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.fc4(x)
        x = self.relu_fc4(x)
        output = self.out(x)
        return output

# ----------------------------


class CNN3_Hidden512(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_Hidden512, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # 第一个全连接层的输出扩大到512
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)  # 第二个全连接层的输出为256
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 128)  # 第三个全连接层的输出为128
        self.relu_fc3 = nn.ReLU()

        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output

# ----------------------------


class CNN3_Hidden128(nn.Module):
    def __init__(self, num_classes: int):
        super(CNN3_Hidden128, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 第一个全连接层的输出缩小到128
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)  # 第二个全连接层的输出进一步减少到64
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 32)  # 第三个全连接层的输出为32
        self.relu_fc3 = nn.ReLU()

        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        output = self.out(x)
        return output

# ----------------------------


class CNN3_TwoDropoutLayers(nn.Module):
    def __init__(self, num_classes: int, dropout_rate=0.5):
        super(CNN3_TwoDropoutLayers, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # 第一个Dropout层

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # 第二个Dropout层

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.dropout2(x)
        output = self.out(x)
        return output

# ----------------------------


class CNN3_ThreeDropoutLayers(nn.Module):
    def __init__(self, num_classes: int, dropout_rate=0.5):
        super(CNN3_ThreeDropoutLayers, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # 第一个全连接层后的Dropout

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # 第二个全连接层后的Dropout

        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)  # 第三个全连接层后的Dropout

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.dropout3(x)
        output = self.out(x)
        return output


# ----------------------------

class CNN3_FourDropoutLayers(nn.Module):
    def __init__(self, num_classes: int, dropout_rate=0.5):
        super(CNN3_FourDropoutLayers, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)  # 在卷积层后添加Dropout
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)  # 在另一个卷积层后添加Dropout
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # 第一个全连接层后的Dropout

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # 第二个全连接层后的Dropout

        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)  # 第三个全连接层后的Dropout

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.dropout3(x)
        output = self.out(x)
        return output

# ----------------------------


class CNN3_Dropout20(nn.Module):
    def __init__(self, num_classes: int, dropout_rate=0.2):
        super(CNN3_Dropout20, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # 第一个Dropout层

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # 第二个Dropout层

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.dropout2(x)
        output = self.out(x)
        return output

# ----------------------------


class CNN3_Dropout40(nn.Module):
    def __init__(self, num_classes: int, dropout_rate=0.4):
        super(CNN3_Dropout40, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # 第一个Dropout层

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # 第二个Dropout层

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.dropout2(x)
        output = self.out(x)
        return output

# ----------------------------


class CNN3_Dropout60(nn.Module):
    def __init__(self, num_classes: int, dropout_rate=0.6):
        super(CNN3_Dropout60, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # 第一个Dropout层

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 64)
        self.relu_fc3 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # 第二个Dropout层

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.dropout2(x)
        output = self.out(x)
        return output
