import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import torchvision.models as models
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from datetime import datetime

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

def create_training_folder(base_path="checkpoints"):
    '''
    增量式保存模型的文件夹
    '''
    # 获取当前时间并生成文件夹名称
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"training_{time_stamp}"
    folder_path = os.path.join(base_path, folder_name)

    # 如果文件夹不存在，则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f"Training folder created: {folder_path}")
    return folder_path


def resume_training(model, optimizer, checkpoint_path, scheduler=None):
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed training from epoch {start_epoch}, loaded weights from {checkpoint_path}")
    else:
        print("No checkpoint found. Training from scratch.")
    return start_epoch


def train_model(model, criterion, optimizer, num_epochs, save_interval, base_save_path,
                train_loader,
                val_loader,
                device,
                scheduler=None, 
                resume_checkpoint=None):
    '''
    训练模型
    '''
    best_val_accuracy = 0.0
    save_folder = create_training_folder(base_save_path)  # 创建保存模型的文件夹

    # 如果提供了恢复模型的路径，加载模型和优化器状态
    start_epoch = 0
    if resume_checkpoint:
        start_epoch = resume_training(model, optimizer, resume_checkpoint, scheduler)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", ncols=100):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", ncols=100, leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_model.pth'))
            print(f"Best model saved with accuracy: {val_accuracy:.4f}")

        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None
            }
            torch.save(checkpoint, os.path.join(save_folder, f"model_epoch_{epoch+1}.pth"))
            print(f"Checkpoint saved at epoch {epoch+1}")


# 测试函数
def evaluate_model(model, test_loader, device):
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Testing", ncols=100, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()

    test_accuracy = test_correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

'''
可视化并保存预测结果，显示多个图片
'''
def visualize_predictions(model, transform, folder_path, save_path, num_images, device):
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'jpeg'))]
    selected_images = image_files[:num_images]  # 只选择前 num_images 张图片

    # 设置可视化的行列数
    num_cols = 5  # 每行显示5张图片
    num_rows = (num_images + 1) // num_cols  # 计算行数

    # 设置绘图区域
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 5))

    if num_rows == 1:
        axes = [axes]  # 如果只有一行，确保 axes 是列表

    # 遍历并显示每张图片及其预测结果
    for i, img_name in enumerate(selected_images):
        img_path = os.path.join(folder_path, img_name)

        # 加载图像但不进行transform
        image = Image.open(img_path).convert('RGB')

        # 仅在模型预测时应用transform
        image_tensor = transform(image).unsqueeze(0).to(device)  # 增加批次维度并移动到设备上

        # 模型预测
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # 获取真实标签（假设背景为0，UAV为1）
        true_label = 0 if 'background' in folder_path else 1  # 根据文件名判断真实标签

        # 显示图像及其预测
        ax = axes[i // num_cols][i % num_cols]  # 获取当前子图的位置
        image = np.array(image)  # 将原始图像转换为numpy数组以便显示
        ax.imshow(image)
        ax.set_title(f"True: {true_label}, Pred: {predicted.item()}")
        ax.axis('off')  # 不显示坐标轴

    # 调整子图间距
    plt.tight_layout()

    # 保存整个图像
    plt.savefig(save_path)
    print(f"Prediction results saved to {save_path}")

    plt.show()  # 显示图像


# 主程序
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义数据集路径
    train_dir = '/qirui.hu1/UAV-dataset/dataset1/train'
    val_dir = '/qirui.hu1/UAV-dataset/dataset1/val'
    test_dir = '/qirui.hu1/UAV-dataset/dataset1/test'

    # 数据增强和转换（包括图像增强）
    transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # 随机裁剪并缩放到64x64
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转，最大旋转角度15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整亮度、对比度、饱和度和色调
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化，使用ImageNet预训练模型的标准均值和标准差
    ])

    # test数据仅进行resize和归一化,删除随机操作
    transform_raw = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 创建数据集和数据加载器
    train_dataset = ImageDataset(os.path.join(train_dir, 'UAV'), os.path.join(train_dir, 'background'), transform=transform)
    val_dataset = ImageDataset(os.path.join(val_dir, 'UAV'), os.path.join(val_dir, 'background'), transform=transform)
    test_dataset = ImageDataset(os.path.join(test_dir, 'UAV'), os.path.join(test_dir, 'background'), transform=transform_raw)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    # 加载预训练的ResNet18模型
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 调整全连接层用于二分类
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500
    root_path = "/qirui.hu1/UAV-dataset/"
    base_save_path = root_path + 'checkpoints/'
    pred_path1 = root_path + "dataset1/test/UAV/"
    pred_path2 = root_path + "dataset1/test/background/"
    save_path = root_path + "visualize/"

    best_model_path = root_path + "checkpoints/training_2024-12-21_20-52-39/best_model.pth"
    # 训练模型
    # train_model(model, criterion, optimizer, num_epochs, save_interval=50, base_save_path=base_save_path, train_loader = train_loader, val_loader = val_loader, device = device)
    
    # 加载最优模型权重进行测试
    checkpoint = torch.load(best_model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])  # model_epoch_500.pth
    model.load_state_dict(checkpoint)
    evaluate_model(model, test_loader, device)
    # 可视化
    visualize_predictions(model, transform, pred_path1, save_path + "pred1.png", 10, device)
    visualize_predictions(model, transform, pred_path2, save_path + "pred2.png", 10, device)

if __name__ == "__main__":
    main()