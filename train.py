import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import torchmetrics
from model import Autoencoder, CNN

device = "cuda" if torch.cuda.is_available() else "cpu"
print('[INFO] Running on ' + device)

# 超参数定义
TRAIN_BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

# 数据预处理与加载
def getDataLoader(data_dir):
    # 创建数据增强器
    transform = transforms.Compose([
        transforms.Grayscale(),  # 转灰度图
        transforms.Resize((256, 256)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为tensor
    ])
    # 构建数据集
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # 构建数据加载器
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    return dataset, dataloader

# 给图像添加噪声
def add_noise(inputs, noise_factor=0.5):
    noisy_inputs = inputs + noise_factor * torch.randn_like(inputs).to(device)
    noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)  # 确保图片值在[0, 1]范围内
    return noisy_inputs

# 模型训练
def train_autoencoder_process(train_dataloader, autoencoder_model):
    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    loss = []
    for epoch in range(EPOCHS):
        loss_epoch = 0
        for imgs, labels in tqdm(train_dataloader, desc=f"[Autoencoder] Epoch {epoch + 1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            noisy_imgs = add_noise(imgs)
            denoise_imgs = autoencoder_model(noisy_imgs)
            train_loss = loss_func(denoise_imgs, imgs)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            loss_epoch += train_loss.item()
        epoch_avg_loss = loss_epoch / len(train_dataloader)
        print("[Autoencoder] Epoch:", epoch + 1, "--Loss:", epoch_avg_loss)
        loss.append(epoch_avg_loss)
    return loss

def train_cnn_process(train_dataloader, trained_autoencoder_model, cnn_model):
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    loss = []
    accuracy = []
    for epoch in range(EPOCHS):
        loss_epoch = 0
        test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=3).to(device)
        for imgs, labels in tqdm(train_dataloader, desc=f"[CNN] Epoch {epoch + 1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            noisy_imgs = add_noise(imgs)
            with torch.no_grad():
                denoise_imgs = trained_autoencoder_model(noisy_imgs)
            y_pred = cnn_model(denoise_imgs)
            train_loss = loss_func(y_pred, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            loss_epoch += train_loss.item()
            test_acc.update(y_pred, labels)
        epoch_avg_loss = loss_epoch / len(train_dataloader)
        epoch_accuracy = 100 * float(test_acc.compute())
        print("[CNN] Epoch:", epoch + 1, "--Loss:", epoch_avg_loss, f"--Accuracy: {epoch_accuracy:.2f}%")
        loss.append(epoch_avg_loss)
        accuracy.append(epoch_accuracy)
    return loss, accuracy

def data_visualize(epoch, data, y_label, title, save_path):
    plt.plot(epoch, data)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(title)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis="y", linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path)  # 保存图像
    plt.close()  # 关闭图形，以释放资源
            
train_data_dir = 'DataSet/train'
train_dataset, train_dataloader = getDataLoader(train_data_dir)
epoch = range(1, EPOCHS + 1)

autoencoder = Autoencoder().to(device)
autoencoder_train_loss = train_autoencoder_process(train_dataloader, autoencoder)
torch.save(autoencoder.state_dict(), 'Trained_Model/autoencoder.pth')
data_visualize(epoch, autoencoder_train_loss, 'Loss', 'Autoencoder_Model Train Loss', 'Autoencoder_Model_Train_Loss.png')

cnn = CNN().to(device)
cnn_train_loss, cnn_train_accuracy = train_cnn_process(train_dataloader, autoencoder, cnn)
torch.save(cnn.state_dict(), 'Trained_Model/cnn.pth')
data_visualize(epoch, cnn_train_loss, 'Loss', 'CNN_Model Train Loss', 'CNN_Model_Train_Loss.png')
data_visualize(epoch, cnn_train_accuracy, 'Accuracy(%)', 'CNN_Model Accuracy', 'CNN_Model_Accuracy.png')