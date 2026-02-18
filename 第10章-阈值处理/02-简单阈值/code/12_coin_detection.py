"""
使用阈值处理检测硬币
结合阈值处理、形态学操作和轮廓检测
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 硬币检测函数 =====================

def detect_coins(image):
    """使用阈值处理检测图像中的硬币"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 使用Otsu自动阈值
    ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu自动选择的阈值: {ret}")

    # 形态学操作：填充小洞并去除小噪点
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选圆形轮廓（硬币）
    coins = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        # 计算圆度
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            coins.append((int(x), int(y), int(radius)))

    # 在原图上绘制结果
    if len(image.shape) == 3:
        result = image.copy()
    else:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for (x, y, r) in coins:
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)

    return binary, result, len(coins)

# ===================== 创建模拟硬币图像 =====================

def create_coin_image():
    img = np.ones((400, 600), dtype=np.uint8) * 40  # 深色背景

    coins_pos = [(100, 150, 50), (250, 200, 45), (400, 120, 55),
                 (180, 320, 40), (450, 300, 50)]

    for x, y, r in coins_pos:
        cv2.circle(img, (x, y), r, 200, -1)
        cv2.circle(img, (x, y), r-5, 220, 2)

    # 添加噪声
    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 检测硬币 =====================

coin_img = create_coin_image()
binary, result, count = detect_coins(coin_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(coin_img, cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('二值化掩码 (Otsu)', fontsize=12)
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'检测结果: {count} 个硬币', fontsize=12)
axes[2].axis('off')

plt.suptitle('使用阈值处理检测硬币', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('coin_detection.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n检测结果：发现 {count} 个硬币")
