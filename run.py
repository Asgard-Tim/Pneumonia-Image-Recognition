import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from flask import Flask, request, render_template, redirect
from torchvision import transforms
from werkzeug.utils import secure_filename
from model import Autoencoder, CNN

app = Flask(__name__)

# 配置文件
app.config['UPLOAD_FOLDER'] = 'static/uploaded'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大上传文件大小 16MB

# 初始化设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print('[INFO] Running on ' + device)

# 加载训练好的模型
autoencoder = Autoencoder().to(device)
autoencoder.load_state_dict(torch.load('Trained_Model/autoencoder.pth', weights_only=False))
cnn = CNN().to(device)
cnn.load_state_dict(torch.load('Trained_Model/cnn.pth', weights_only=False))

# 检查文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 上传图片并进行预测
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert('RGB')
    img.convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 添加噪声并进行去噪
    noisy_img = img_tensor + 0.2 * torch.randn_like(img_tensor).to(device)
    noisy_img = torch.clamp(noisy_img, 0., 1.)

    with torch.no_grad():
        denoised_img = autoencoder(noisy_img)

    # 使用CNN进行分类
    with torch.no_grad():
        output = cnn(denoised_img)
        _, predicted = torch.max(output, 1)

    # 返回分类结果
    class_names = ['COVID19', 'NORMAL', 'PNEUMONIA']
    class_label = class_names[predicted.item()]

    # 将去噪后的图像转换为 NumPy 数组，确保它是二维的
    denoised_img = denoised_img.squeeze(0).cpu().numpy()  # 去除批次维度
    if denoised_img.shape[0] == 1:
        denoised_img = denoised_img[0]  # 只有一个通道，取第一个通道

    # 保存去噪后的图像
    denoised_img_path = os.path.join(app.config['RESULT_FOLDER'], 'denoised.png')
    plt.imsave(denoised_img_path, denoised_img, cmap='gray')

    return class_label, denoised_img_path

# 首页接口，允许用户上传图片
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取上传的文件
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 处理图片并获取分类结果
            class_label, denoised_img_path = process_image(filepath)

            return render_template('index.html',
                                   filename=filename,
                                   class_label=class_label,
                                   denoised_img_path=denoised_img_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)