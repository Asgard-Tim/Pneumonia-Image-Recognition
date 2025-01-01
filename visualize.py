import torch
from torchviz import make_dot
from torchsummary import summary
from model import Autoencoder, CNN
from torchvision import transforms
from PIL import Image

# 加载图片并处理
image_path = 'examples/1.jpeg'  # 替换为你的图片路径
image = Image.open(image_path).convert('L')  # 转换为灰度图

# 预处理图片：调整大小并转换为PyTorch张量
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图片大小为256x256
    transforms.ToTensor(),  # 转换为Tensor，并将像素值缩放到[0, 1]范围
])

device = "cuda" if torch.cuda.is_available() else "cpu"
print('[INFO] Running on ' + device)

autoencoder = Autoencoder().to(device)
cnn = CNN().to(device)

x = transform(image).to(device).unsqueeze(0)  # 增加一个批次维度，变成 (1, 1, 256, 256)
y = autoencoder(x)

# 使用torchviz生成图形
dot = make_dot(y, params=dict(autoencoder.named_parameters()))
dot.render("autoencoder_architecture", format="png")  # 将结果保存为PNG图像

# 使用torchsummary生成模型概况
summary(autoencoder, (1, 256, 256))

y = cnn(x)

# 使用torchviz生成图形
dot = make_dot(y, params=dict(autoencoder.named_parameters()))
dot.render("cnn_architecture", format="png")  # 将结果保存为PNG图像

# 使用torchsummary生成模型概况
summary(cnn, (1, 256, 256))