"""
示例8：边界模式对比
- BORDER_CONSTANT / REPLICATE / REFLECT / REFLECT_101 / WRAP
- 变换后越界像素的不同处理方式
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建渐变测试图像
image = np.zeros((150, 200, 3), dtype=np.uint8)
for i in range(200):
    image[:, i] = [i, 100, 200 - i]
cv2.rectangle(image, (30, 30), (170, 120), (255, 255, 255), 2)
cv2.putText(image, 'BORDER', (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

h, w = image.shape[:2]

# 变换（旋转+平移，使部分图像超出边界）
M = cv2.getRotationMatrix2D((w // 2, h // 2), 20, 1.0)
M[0, 2] += 60
M[1, 2] += 40

# 各种边界模式
border_modes = {
    'CONSTANT (黑色)': (cv2.BORDER_CONSTANT, (0, 0, 0)),
    'CONSTANT (红色)': (cv2.BORDER_CONSTANT, (0, 0, 255)),
    'REPLICATE': (cv2.BORDER_REPLICATE, None),
    'REFLECT': (cv2.BORDER_REFLECT, None),
    'REFLECT_101': (cv2.BORDER_REFLECT_101, None),
    'WRAP': (cv2.BORDER_WRAP, None),
}

results = {}
for name, (mode, value) in border_modes.items():
    if value is not None:
        results[name] = cv2.warpAffine(image, M, (w, h), borderMode=mode, borderValue=value)
    else:
        results[name] = cv2.warpAffine(image, M, (w, h), borderMode=mode)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('边界模式对比 (borderMode)', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

for i, (name, result) in enumerate(results.items()):
    row, col = (i + 1) // 4, (i + 1) % 4
    axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(name)
    axes[row, col].axis('off')

# 说明文字
explanation = """边界模式说明:

CONSTANT:   用常数填充
            aaa|abcd|aaa

REPLICATE:  复制边缘像素
            aaa|abcd|ddd

REFLECT:    镜像反射
            dcb|abcd|dcba

REFLECT_101:不含边界的反射
            dcb|abcdefg|fedc

WRAP:       环绕（周期）
            bcd|abcd|abcd"""
axes[1, 3].text(0.05, 0.5, explanation, fontsize=8, family='monospace',
                verticalalignment='center', transform=axes[1, 3].transAxes)
axes[1, 3].axis('off')
axes[1, 3].set_title('边界模式说明')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_border_modes.png'), dpi=150, bbox_inches='tight')
plt.show()

print("边界模式用法:")
print("  cv2.warpAffine(img, M, size, borderMode=cv2.BORDER_CONSTANT, borderValue=(B,G,R))")
