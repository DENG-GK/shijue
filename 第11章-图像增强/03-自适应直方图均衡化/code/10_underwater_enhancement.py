"""
示例10：水下图像增强
- 白平衡校正消除色偏
- CLAHE增强对比度（LAB空间）
- 轻微锐化突出细节
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def enhance_underwater_image(image):
    """水下图像增强"""
    # 1. 白平衡校正
    result = image.copy().astype(np.float32)

    for i in range(3):
        avg = result[:, :, i].mean()
        if avg > 0:
            result[:, :, i] = result[:, :, i] * (128 / avg)

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 2. CLAHE增强（LAB空间）
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3. 轻微锐化
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5, -0.5],
                       [-0.5, -0.5, -0.5]])
    result = cv2.filter2D(result, -1, kernel)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def create_underwater_image():
    """创建模拟水下图像"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    base_b, base_g, base_r = 120, 100, 60

    for i in range(300):
        for j in range(400):
            variation = int(20 * np.sin(i / 30) * np.cos(j / 40))
            img[i, j] = [base_b + variation,
                        base_g + variation - 10,
                        base_r + variation - 20]

    cv2.circle(img, (100, 150), 30, (100, 80, 50), -1)
    cv2.ellipse(img, (280, 120), (40, 20), 20, 0, 360, (110, 90, 55), -1)
    cv2.rectangle(img, (150, 200), (250, 270), (90, 75, 45), -1)

    # 添加雾化效果
    fog = np.ones_like(img) * np.array([100, 90, 70])
    img = cv2.addWeighted(img.astype(np.float32), 0.7, fog.astype(np.float32), 0.3, 0)
    img = img.astype(np.uint8)

    return img


underwater = create_underwater_image()
enhanced = enhance_underwater_image(underwater)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('水下图像增强', fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(underwater, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始水下图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
axes[1].set_title('增强后（白平衡 + CLAHE + 锐化）', fontsize=12)
axes[1].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_underwater_enhancement.png'), dpi=150, bbox_inches='tight')
plt.show()

print("水下图像增强流程：")
print("1. 白平衡校正（消除色偏）")
print("2. CLAHE对比度增强（LAB空间）")
print("3. 锐化（可选）")
