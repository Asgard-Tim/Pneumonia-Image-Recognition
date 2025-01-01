import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchmetrics
from model import Autoencoder, CNN

device = "cuda" if torch.cuda.is_available() else "cpu"
print('[INFO] Running on ' + device)

TEST_BATCH_SIZE = 66

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
    dataloader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)
    return dataset, dataloader

# 给图像添加噪声
def add_noise(inputs, noise_factor=0.5):
    noisy_inputs = inputs + noise_factor * torch.randn_like(inputs).to(device)
    noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)  # 确保图片值在[0, 1]范围内
    return noisy_inputs

def with_noise_test(with_noise_test_dataloader, autoencoder_model, cnn_model):
    test_acc = torchmetrics.Accuracy(task='multiclass',num_classes=3).to(device)
    with torch.no_grad():
        for imgs, labels in with_noise_test_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            denoise_imgs = autoencoder_model(imgs)
            y_pred = cnn_model(denoise_imgs)
            test_acc.update(y_pred, labels)
    accuracy = test_acc.compute()
    return accuracy

def without_noise_test(without_noise_test_dataloader, autoencoder_model, cnn_model):
    test_acc = torchmetrics.Accuracy(task='multiclass',num_classes=3).to(device)
    with torch.no_grad():
        for imgs, labels in without_noise_test_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            noise_imgs = add_noise(imgs)
            denoise_imgs = autoencoder_model(noise_imgs)
            y_pred = cnn_model(denoise_imgs)
            test_acc.update(y_pred, labels)
    accuracy = test_acc.compute()
    return accuracy

# 绘制去噪图像
def plot_denoising(autoencoder, noisy_image, clean_image):
    autoencoder.eval()
    with torch.no_grad():
        denoised_image = autoencoder(noisy_image.unsqueeze(0).to(device))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(noisy_image.cpu().squeeze(), cmap='gray')
    ax[0].set_title('Noisy Image')
    ax[1].imshow(clean_image.cpu().squeeze(), cmap='gray')
    ax[1].set_title('Clean Image')
    ax[2].imshow(denoised_image.cpu().squeeze(), cmap='gray')
    ax[2].set_title('Denoised Image')
    plt.show()

# 加载训练好的模型
autoencoder = Autoencoder().to(device)
autoencoder.load_state_dict(torch.load('Trained_Model/autoencoder.pth', weights_only=False))
cnn = CNN().to(device)
cnn.load_state_dict(torch.load('Trained_Model/cnn.pth', weights_only=False))

# 有噪声测试集
with_noise_test_data_dir = 'DataSet/noisy_test'
with_noise_test_dataset, with_noise_test_dataloader = getDataLoader(with_noise_test_data_dir)
with_noise_test_accuracy = with_noise_test(with_noise_test_dataloader, autoencoder, cnn)
print(f"With Noise Test Accuracy: {100*float(with_noise_test_accuracy):.2f}%")

# 无噪声测试集
without_noise_test_data_dir = 'DataSet/test'
without_noise_test_dataset, without_noise_test_dataloader = getDataLoader(without_noise_test_data_dir)
without_noise_test_accuracy = without_noise_test(without_noise_test_dataloader, autoencoder, cnn)
print(f"Without Noise Test Accuracy: {100*float(without_noise_test_accuracy):.2f}%")

# 去噪结果可视化
clean_img, _ = without_noise_test_dataset[0]  # 获取一个无噪声样本
clean_img = clean_img.to(device)
noisy_img = add_noise(clean_img, noise_factor=0.5)  # 对原始图像加噪
plot_denoising(autoencoder, noisy_img, clean_img)