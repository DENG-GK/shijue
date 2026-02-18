"""
示例6：预处理对轮廓检测的影响
- 高斯模糊 / 中值滤波
- 不同二值化方法
- 形态学操作
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建带噪声的图像
np.random.seed(42)
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (30, 30), (180, 150), 200, -1)
cv2.circle(img, (300, 100), 60, 200, -1)
cv2.ellipse(img, (120, 230), (70, 40), 0, 0, 360, 200, -1)
cv2.rectangle(img, (260, 180), (380, 280), 200, -1)

# 添加噪声
noise = np.random.normal(0, 25, img.shape).astype(np.int16)
noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 不同预处理方法
preprocessed = {
    '无预处理': noisy,
    '高斯模糊 5x5': cv2.GaussianBlur(noisy, (5, 5), 0),
    '中值滤波 5': cv2.medianBlur(noisy, 5),
    '高斯 + 形态学': None,
}

# 高斯 + 形态学
blurred = cv2.GaussianBlur(noisy, (5, 5), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
preprocessed['高斯 + 形态学'] = morph

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('预处理对轮廓检测的影响', fontsize=14, fontweight='bold')

for idx, (name, processed) in enumerate(preprocessed.items()):
    _, binary = cv2.threshold(processed, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    axes[0, idx].imshow(processed, cmap='gray')
    axes[0, idx].set_title(name)
    axes[0, idx].axis('off')

    canvas = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    axes[1, idx].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[1, idx].set_title(f'{len(contours)} 个轮廓')
    axes[1, idx].axis('off')

    print(f"{name}: {len(contours)} 个轮廓")

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_preprocessing.png'), dpi=150, bbox_inches='tight')
plt.show()
