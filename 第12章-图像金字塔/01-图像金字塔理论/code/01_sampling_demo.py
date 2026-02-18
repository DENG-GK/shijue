"""
示例1：理解下采样和上采样
- pyrDown：高斯滤波 + 隔行隔列删除
- pyrUp：插入零值 + 高斯滤波
- 多级下采样/上采样的信息损失
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((256, 256, 3), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (200, 200), (0, 255, 0), -1)
cv2.circle(image, (128, 128), 50, (255, 0, 0), -1)
cv2.putText(image, 'Test', (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

print(f"原始图像尺寸: {image.shape}")

# 多级下采样
down1 = cv2.pyrDown(image)
down2 = cv2.pyrDown(down1)
down3 = cv2.pyrDown(down2)

# 从最小尺寸多级上采样
up1 = cv2.pyrUp(down3)
up2 = cv2.pyrUp(up1)
up3 = cv2.pyrUp(up2)

print(f"pyrDown 1: {down1.shape}")
print(f"pyrDown 2: {down2.shape}")
print(f"pyrDown 3: {down3.shape}")
print(f"pyrUp 回 1: {up1.shape}")
print(f"pyrUp 回 2: {up2.shape}")
print(f"pyrUp 回 3: {up3.shape}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('下采样与上采样', fontsize=14, fontweight='bold')

# 下采样
for i, (img, title) in enumerate([
    (image, f'原图\n{image.shape[:2]}'),
    (down1, f'pyrDown×1\n{down1.shape[:2]}'),
    (down2, f'pyrDown×2\n{down2.shape[:2]}'),
    (down3, f'pyrDown×3\n{down3.shape[:2]}'),
]):
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(title)
    axes[0, i].axis('off')

# 上采样
for i, (img, title) in enumerate([
    (down3, f'最小\n{down3.shape[:2]}'),
    (up1, f'pyrUp×1\n{up1.shape[:2]}'),
    (up2, f'pyrUp×2\n{up2.shape[:2]}'),
    (up3, f'pyrUp×3\n{up3.shape[:2]}'),
]):
    axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(title)
    axes[1, i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_sampling_demo.png'), dpi=150, bbox_inches='tight')
plt.show()

# 重建误差
if up3.shape == image.shape:
    diff = cv2.absdiff(image, up3)
    print(f"\n重建误差（均值）: {np.mean(diff):.2f}")
    print("注意：下采样再上采样会丢失高频信息，无法完美恢复")
