"""
示例3：图像翻转
- cv2.flip() 三种模式
- flipCode=1 水平翻转
- flipCode=0 垂直翻转
- flipCode=-1 双轴翻转(180°旋转)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建不对称测试图像
image = np.zeros((200, 300, 3), dtype=np.uint8)
image[:, :] = [200, 200, 200]

# 添加不对称元素
cv2.putText(image, 'F', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)
cv2.circle(image, (220, 60), 30, (0, 255, 0), -1)
cv2.rectangle(image, (180, 120), (280, 180), (0, 0, 255), -1)

# 翻转操作
flip_horizontal = cv2.flip(image, 1)    # flipCode > 0: 水平翻转
flip_vertical = cv2.flip(image, 0)      # flipCode = 0: 垂直翻转
flip_both = cv2.flip(image, -1)         # flipCode < 0: 双轴翻转

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('图像翻转操作 cv2.flip()', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(flip_horizontal, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('水平翻转 (flipCode=1)')
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(flip_vertical, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('垂直翻转 (flipCode=0)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(flip_both, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('双轴翻转 (flipCode=-1)')
axes[1, 1].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_flip_demo.png'), dpi=150, bbox_inches='tight')
plt.show()

print("翻转操作说明:")
print("  cv2.flip(img, 1)  → 水平翻转（左右镜像）")
print("  cv2.flip(img, 0)  → 垂直翻转（上下镜像）")
print("  cv2.flip(img, -1) → 双轴翻转（等效180°旋转）")
