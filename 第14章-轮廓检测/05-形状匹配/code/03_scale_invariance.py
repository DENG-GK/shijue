"""
示例3：缩放不变性验证
- matchShapes 对缩放的不变性测试
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_scaled_circle(radius):
    size = max(100, radius * 3)
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, 255, -1)
    return img


# 参考圆
ref_img = create_scaled_circle(50)
ref_cnt, _ = cv2.findContours(ref_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ref_cnt = ref_cnt[0]

radii = [20, 30, 40, 50, 60, 80, 100, 120]

print("缩放不变性测试 (圆形):")
print("-" * 50)
print(f"{'半径':>8} {'面积':>10} {'I1':>12} {'I2':>12}")
print("-" * 50)

results = []
for radius in radii:
    img = create_scaled_circle(radius)
    cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(cnt[0])
    s1 = cv2.matchShapes(ref_cnt, cnt[0], cv2.CONTOURS_MATCH_I1, 0)
    s2 = cv2.matchShapes(ref_cnt, cnt[0], cv2.CONTOURS_MATCH_I2, 0)
    print(f"{radius:>8} {area:>10.0f} {s1:>12.6f} {s2:>12.6f}")
    results.append((radius, img, s1))

print("\n结论: Hu矩匹配对缩放具有不变性！")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('缩放不变性验证 (圆形)', fontsize=14, fontweight='bold')

for idx, (radius, img, score) in enumerate(results):
    r, c = idx // 4, idx % 4
    axes[r, c].imshow(img, cmap='gray')
    axes[r, c].set_title(f'r={radius}  I1={score:.6f}')
    axes[r, c].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_scale_invariance.png'), dpi=150, bbox_inches='tight')
plt.show()
