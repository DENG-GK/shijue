"""
示例1：基本透视变换
- getPerspectiveTransform 四点定义
- warpPerspective 应用变换
- 上窄/下窄/左倾/右倾/鸟瞰效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((400, 500, 3), dtype=np.uint8)
image[:, :] = [220, 220, 220]
cv2.rectangle(image, (50, 50), (450, 350), (0, 128, 255), -1)
cv2.putText(image, 'PERSPECTIVE', (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

# 标记四角
corners = [(50, 50), (450, 50), (450, 350), (50, 350)]
for i, c in enumerate(corners):
    cv2.circle(image, c, 10, (255, 0, 0), -1)
    cv2.putText(image, str(i + 1), (c[0] + 15, c[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

h, w = image.shape[:2]
src_pts = np.float32([[50, 50], [450, 50], [450, 350], [50, 350]])

transforms = {
    '原始': src_pts,
    '上窄': np.float32([[100, 50], [400, 50], [450, 350], [50, 350]]),
    '下窄': np.float32([[50, 50], [450, 50], [400, 350], [100, 350]]),
    '左倾': np.float32([[50, 80], [450, 30], [450, 320], [50, 370]]),
    '右倾': np.float32([[50, 30], [450, 80], [450, 370], [50, 320]]),
    '鸟瞰': np.float32([[100, 0], [400, 0], [500, 400], [0, 400]]),
}

results = {}
for name, dst_pts in transforms.items():
    if name == '原始':
        results[name] = image.copy()
    else:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        results[name] = cv2.warpPerspective(image, M, (w, h))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('基本透视变换效果', fontsize=14, fontweight='bold')

for i, (name, result) in enumerate(results.items()):
    row, col = i // 3, i % 3
    axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(name)
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_basic_perspective.png'), dpi=150, bbox_inches='tight')
plt.show()

print("基本透视变换完成！")
print("核心函数: cv2.getPerspectiveTransform(src_pts, dst_pts)")
print("         cv2.warpPerspective(image, M, (w, h))")
