import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16  # 加载预训练的 ViT 模型

# 定义数据集路径
train_dir = './dataset1/train'
val_dir = './dataset1/val'
test_dir = './dataset1/test'

# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, uav_dir, background_dir, transform=None):
        self.uav_images = [os.path.join(uav_dir, img) for img in os.listdir(uav_dir)]
        self.background_images = [os.path.join(background_dir, img) for img in os.listdir(background_dir)]
        self.images = self.uav_images + self.background_images
        self.labels = [1] * len(self.uav_images) + [0] * len(self.background_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# 数据增强和转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 输入大小为 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化
])

# 创建数据集和数据加载器
train_dataset = ImageDataset(os.path.join(train_dir, 'UAV'), os.path.join(train_dir, 'background'), transform=transform)
val_dataset = ImageDataset(os.path.join(val_dir, 'UAV'), os.path.join(val_dir, 'background'), transform=transform)
test_dataset = ImageDataset(os.path.join(test_dir, 'UAV'), os.path.join(test_dir, 'background'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

# 定义ViT分类器
class UAVClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(UAVClassifier, self).__init__()
        self.model = vit_b_16(pretrained=True)  # 加载 ViT Base 模型，使用 16x16 Patch
        # 替换最后的分类头部
        self.model.heads = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型、损失函数和优化器
model = UAVClassifier(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 打印每个 epoch 的结果
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {100 * correct/total:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {100 * val_correct/val_total:.2f}%")

# 启动训练
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# 测试函数
def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 可视化部分样本
            for i in range(5):  # 显示部分样本
                img = transforms.ToPILImage()(images[i].cpu())
                plt.imshow(img)
                plt.title(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
                plt.show()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 启动测试
test_model(model, test_loader)
