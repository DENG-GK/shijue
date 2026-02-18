"""
示例2：通过三点定义仿射变换
- 3对控制点确定唯一仿射变换
- 移动单个/多个控制点的效果
- 翻转效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建带三角形标记的测试图像
image = np.zeros((300, 400, 3), dtype=np.uint8)
image[:, :] = [200, 200, 200]

pts = np.array([[100, 50], [300, 50], [200, 250]], np.int32)
cv2.fillPoly(image, [pts], (0, 150, 255))

# 标记控制点
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
labels = ['P1', 'P2', 'P3']
for pt, color, label in zip(pts, colors, labels):
    cv2.circle(image, tuple(pt), 10, color, -1)
    cv2.putText(image, label, (pt[0] - 15, pt[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

h, w = image.shape[:2]
src_pts = pts.astype(np.float32)

# 不同的目标点配置
dest_configs = {
    '原始': pts.astype(np.float32),
    '移动P1': np.float32([[150, 80], [300, 50], [200, 250]]),
    '移动P2': np.float32([[100, 50], [350, 100], [200, 250]]),
    '移动P3': np.float32([[100, 50], [300, 50], [150, 200]]),
    '全部移动': np.float32([[50, 100], [350, 80], [180, 280]]),
    '翻转': np.float32([[300, 50], [100, 50], [200, 250]]),
}

results = {'原始': image.copy()}
for name, dst_pts in dest_configs.items():
    if name != '原始':
        M = cv2.getAffineTransform(src_pts, dst_pts)
        result = cv2.warpAffine(image, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(128, 128, 128))
        # 标记目标点
        for pt, color in zip(dst_pts.astype(np.int32), colors):
            cv2.circle(result, tuple(pt), 8, color, 2)
        results[name] = result

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('三点定义仿射变换', fontsize=14, fontweight='bold')

for i, (name, result) in enumerate(results.items()):
    row, col = i // 3, i % 3
    axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(name)
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_three_point_affine.png'), dpi=150, bbox_inches='tight')
plt.show()

print("三点仿射变换完成！")
print("仿射变换由3对对应点唯一确定 (6个自由度)")
