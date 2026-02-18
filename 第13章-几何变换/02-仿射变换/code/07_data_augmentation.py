"""
示例7：数据增强（图像增广）
- 随机仿射变换用于深度学习数据增强
- 随机旋转/缩放/错切/平移组合
- 生成多种变体
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建样本图像
image = np.zeros((200, 200, 3), dtype=np.uint8)
image[:, :] = [100, 150, 200]
cv2.circle(image, (100, 100), 60, (255, 100, 50), -1)
cv2.rectangle(image, (60, 60), (140, 140), (50, 200, 100), 3)
cv2.putText(image, 'A', (75, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

h, w = image.shape[:2]


def random_affine_transform(img, max_rotation=15, max_scale=0.15,
                            max_shear=0.1, max_translate=0.1):
    """随机仿射变换"""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    angle = random.uniform(-max_rotation, max_rotation)
    scale = random.uniform(1 - max_scale, 1 + max_scale)
    shear_x = random.uniform(-max_shear, max_shear)
    shear_y = random.uniform(-max_shear, max_shear)
    tx = random.uniform(-max_translate, max_translate) * w
    ty = random.uniform(-max_translate, max_translate) * h

    # 构建变换: 居中→错切→缩放→旋转→平移回
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    Sh = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=np.float32)
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    rad = np.radians(angle)
    R = np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    T2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float32)

    M = T2 @ R @ S @ Sh @ T1
    return cv2.warpAffine(img, M[:2, :], (w, h), borderMode=cv2.BORDER_REFLECT_101)


# 生成增强样本
random.seed(42)
augmented = [image]
for _ in range(15):
    augmented.append(random_affine_transform(image))
augmented.append(cv2.flip(image, 1))
augmented.append(cv2.flip(image, 0))

fig, axes = plt.subplots(3, 6, figsize=(18, 10))
fig.suptitle('数据增强 - 随机仿射变换', fontsize=14, fontweight='bold')

for i, img in enumerate(augmented[:18]):
    row, col = i // 6, i % 6
    axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if i == 0:
        axes[row, col].set_title('原始')
    elif i == 16:
        axes[row, col].set_title('水平翻转')
    elif i == 17:
        axes[row, col].set_title('垂直翻转')
    else:
        axes[row, col].set_title(f'增强 {i}')
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_data_augmentation.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"数据增强完成！共生成 {len(augmented)} 张图像变体")
