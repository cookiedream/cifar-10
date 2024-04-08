import os
import numpy as np
import matplotlib.pyplot as plt
from dataloader import data_loader, test_loader  # 假設您有一個test_loader
from model import CNN, CNN3, CNN4, ResNet9, CNN3_16, CNN3_32, CNN3_64, CNN3_44, CNN3_77, CNN3_MaxPool3, CNN3_MaxPool4, CNN3_TwoHiddenLayers, CNN3_FourHiddenLayers, CNN3_Hidden512, CNN3_Hidden128, CNN3_TwoDropoutLayers, CNN3_ThreeDropoutLayers, CNN3_FourDropoutLayers, CNN3_Dropout20, CNN3_Dropout40, CNN3_Dropout60  # 假設您有一個CNN模型
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sys  # 導入sys模組
from torchsummary import summary

# 設定參數
LR = 1e-3
EPOCH = 500
MODELS_PATH = './weight'
TENSORBOARD_PATH = "CNN3_50123"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TensorBoard寫入器
writer = SummaryWriter(f'runs/train_{TENSORBOARD_PATH}')

model = CNN3(num_classes=10).to(device)
# model = ResNet9(num_classes=10, in_channels=3).to(device)
if not os.path.exists('./result'):
    os.mkdir('./result')
if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)


class DualOutput:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  # 這個flush方法是為了兼容sys.stdout的flush方法。
        # Flush的作用是清空緩衝區，將緩衝區中的數據立即寫入文件，對於寫入文件操作是非常有用的。
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal  # 恢復標準輸出到控制台


sys.stdout = DualOutput(f'./result/output_{TENSORBOARD_PATH}.txt')
summary(model, input_size=(3, 32, 32))

# 設定優化器和損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

# 用於繪圖的數據存儲
train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []
best_test_acc = 0.0  # 初始最佳準確率設為0
best_model_path = os.path.join(
    MODELS_PATH, f'best_cnn_model_{TENSORBOARD_PATH}.pt')  # 定義最佳模型的儲存路徑
for epoch in range(EPOCH):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for x, y in data_loader:
        b_x = x.to(device)
        b_y = y.to(device)
        out = model(b_x)
        loss = loss_function(out, b_y)
        train_loss += loss.item()
        pred = torch.argmax(out, 1)
        correct_train += (pred == b_y).sum().item()
        total_train += b_y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(train_loss / len(data_loader))
    train_accuracy.append(correct_train / total_train)

    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for x, y in test_loader:
            b_x = x.to(device)
            b_y = y.to(device)
            out = model(b_x)
            loss = loss_function(out, b_y)
            test_loss += loss.item()
            pred = torch.argmax(out, 1)
            correct_test += (pred == b_y).sum().item()
            total_test += b_y.size(0)

    test_losses.append(test_loss / len(test_loader))
    test_accuracy.append(correct_test / total_test)

    # 检查并保存最佳模型
    if test_accuracy[-1] > best_test_acc:
        best_test_acc = test_accuracy[-1]
        torch.save(model.state_dict(), best_model_path)
        print(
            f"New best model saved at epoch {epoch+1} with Test Accuracy: {test_accuracy[-1]:.4f}")
    # 找出最佳的准确率和损失率
    best_train_acc = max(train_accuracy)
    best_test_acc = max(test_accuracy)
    best_train_loss = min(train_losses)
    best_test_loss = min(test_losses)
    # 更新並保存最佳模型
    print(
        f'Epoch {epoch+1}:  Train Accuracy: {train_accuracy[-1]:.4f} Test Accuracy: {test_accuracy[-1]:.4f} Train Loss: {train_losses[-1]:.4f} Test Loss: {test_losses[-1]:.4f}')
    writer.add_scalars(
        'Accuracy', {'Train': train_accuracy[-1], 'Test': test_accuracy[-1]}, epoch+1)
    writer.add_scalars(
        'Loss', {'Train': train_losses[-1], 'Test': test_losses[-1]}, epoch+1)


print(f'Best Train Accuracy: {best_train_acc:.4f} Best Test Accuracy: {best_test_acc:.4f} Best Train Loss: {best_train_loss:.4f} Best Test Loss: {best_test_loss:.4f}')
writer.close()


# 保存模型
torch.save(model.state_dict(), os.path.join(
    MODELS_PATH, f'cnn_model_{TENSORBOARD_PATH}.pt'))


# 程式碼結束，關閉DualOutput並恢復標準輸出
sys.stdout.close()
