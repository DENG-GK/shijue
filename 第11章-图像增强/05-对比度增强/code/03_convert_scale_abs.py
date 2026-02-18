"""
示例3：OpenCV的convertScaleAbs
- 公式：output = alpha * input + beta
- alpha控制对比度，beta控制亮度
- 对比不同alpha和beta组合的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def adjust_contrast_brightness(image, alpha=1.0, beta=0):
    """调整对比度和亮度"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# 创建测试图像
image = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (180, 200), (80, 120, 60), -1)
cv2.circle(image, (300, 150), 60, (150, 80, 100), -1)

params = [
    (1.0, 0, '原图 α=1.0,β=0'),
    (1.5, 0, '高对比度 α=1.5'),
    (0.5, 0, '低对比度 α=0.5'),
    (1.0, 50, '增亮 β=50'),
    (1.0, -50, '减暗 β=-50'),
    (1.3, 30, 'α=1.3,β=30'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('convertScaleAbs 对比度与亮度调整', fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, (alpha, beta, title) in enumerate(params):
    result = adjust_contrast_brightness(image, alpha, beta)
    axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[i].set_title(title, fontsize=11)
    axes[i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_convert_scale_abs.png'), dpi=150, bbox_inches='tight')
plt.show()

print("参数说明：")
print("- alpha > 1: 增加对比度")
print("- alpha < 1: 降低对比度")
print("- beta > 0: 增加亮度")
print("- beta < 0: 降低亮度")
