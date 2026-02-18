"""
手写数字识别预处理
使用自适应阈值处理手写数字，输出MNIST格式(28x28)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 手写数字预处理函数 =====================

def preprocess_handwritten(image, target_size=(28, 28)):
    """
    手写数字图像预处理
    适用于MNIST风格的分类任务
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值（反向，使数字为白色）
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 查找轮廓以定位数字
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找最大轮廓（假设是数字）
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # 裁剪并居中
        digit = binary[y:y+h, x:x+w]

        # 添加边距
        pad = 4
        digit_padded = cv2.copyMakeBorder(digit, pad, pad, pad, pad,
                                          cv2.BORDER_CONSTANT, value=0)

        # 调整大小
        resized = cv2.resize(digit_padded, target_size, interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)

    return binary, resized

# ===================== 创建模拟手写数字 =====================

def create_handwritten_digit():
    img = np.ones((150, 150), dtype=np.uint8) * 230

    # 添加不均匀背景
    for i in range(150):
        for j in range(150):
            factor = 0.8 + 0.2 * np.random.random()
            img[i, j] = int(img[i, j] * factor)

    # 绘制一个手写风格的"5"
    pts = np.array([
        [90, 30], [50, 30], [45, 50], [50, 70], [80, 75],
        [95, 90], [90, 115], [60, 120], [40, 110]
    ], np.int32)

    for i in range(len(pts) - 1):
        p1 = pts[i] + np.random.randint(-2, 3, 2)
        p2 = pts[i+1] + np.random.randint(-2, 3, 2)
        cv2.line(img, tuple(p1), tuple(p2), 40, 8)

    # 添加噪声
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 处理手写数字 =====================

digit_img = create_handwritten_digit()
binary, preprocessed = preprocess_handwritten(digit_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(digit_img, cmap='gray')
axes[0].set_title('原始手写数字\n(背景噪声)', fontsize=11)
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('自适应阈值\n(清理后)', fontsize=11)
axes[1].axis('off')

axes[2].imshow(preprocessed, cmap='gray')
axes[2].set_title('预处理结果 (28x28)\n(可用于分类)', fontsize=11)
axes[2].axis('off')

plt.suptitle('手写数字识别预处理', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('handwritten_digit.png', dpi=150, bbox_inches='tight')
plt.show()

print("手写数字预处理完成！")
print(f"输出尺寸: {preprocessed.shape} (MNIST格式)")
