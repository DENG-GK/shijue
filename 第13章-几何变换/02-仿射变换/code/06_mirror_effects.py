"""
示例6：图像镜像和翻转效果
- 水平/垂直/双轴镜像
- 倒影效果（渐变淡出）
- 万花筒效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建不对称图像
image = np.zeros((200, 300, 3), dtype=np.uint8)
image[:, :] = [220, 200, 180]
cv2.rectangle(image, (30, 30), (150, 170), (0, 128, 255), -1)
cv2.circle(image, (230, 100), 50, (255, 100, 50), -1)
cv2.putText(image, 'Fd', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

h, w = image.shape[:2]

# 各种镜像效果
h_mirror = cv2.flip(image, 1)
v_mirror = cv2.flip(image, 0)
both_mirror = cv2.flip(image, -1)

# 倒影效果
reflection = np.vstack([image, cv2.flip(image, 0)])
for i in range(h):
    alpha = 1 - (i / h) * 0.7
    reflection[h + i, :] = (reflection[h + i, :] * alpha).astype(np.uint8)

# 万花筒效果
top_left = image
top_right = cv2.flip(image, 1)
bottom_left = cv2.flip(image, 0)
bottom_right = cv2.flip(image, -1)
kaleidoscope = np.vstack([np.hstack([top_left, top_right]),
                           np.hstack([bottom_left, bottom_right])])

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('镜像与翻转效果', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(h_mirror, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('水平镜像')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(v_mirror, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('垂直镜像')
axes[0, 2].axis('off')

axes[0, 3].imshow(cv2.cvtColor(both_mirror, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title('180°旋转')
axes[0, 3].axis('off')

axes[1, 0].imshow(cv2.cvtColor(reflection, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('倒影效果')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(kaleidoscope, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('万花筒效果')
axes[1, 1].axis('off')

# 左右对比
comparison = np.hstack([image, h_mirror])
axes[1, 2].imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('原始 | 水平镜像')
axes[1, 2].axis('off')

axes[1, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_mirror_effects.png'), dpi=150, bbox_inches='tight')
plt.show()

print("镜像效果演示完成！")
