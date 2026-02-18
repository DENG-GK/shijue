"""
示例2：旋转不变性验证
- matchShapes 对旋转的不变性测试
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_rotated_square(angle, size=80):
    img = np.zeros((200, 200), dtype=np.uint8)
    half = size // 2
    pts = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float)
    rad = np.radians(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    pts = (pts @ rot.T + [100, 100]).astype(int)
    cv2.fillPoly(img, [pts], 255)
    return img


# 参考形状（0度）
ref_img = create_rotated_square(0)
ref_cnt, _ = cv2.findContours(ref_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ref_cnt = ref_cnt[0]

angles = [0, 15, 30, 45, 60, 90, 120, 180]

print("旋转不变性测试 (正方形):")
print("-" * 55)
print(f"{'角度':>8} {'I1':>12} {'I2':>12} {'I3':>12}")
print("-" * 55)

results = []
for angle in angles:
    img = create_rotated_square(angle)
    cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s1 = cv2.matchShapes(ref_cnt, cnt[0], cv2.CONTOURS_MATCH_I1, 0)
    s2 = cv2.matchShapes(ref_cnt, cnt[0], cv2.CONTOURS_MATCH_I2, 0)
    s3 = cv2.matchShapes(ref_cnt, cnt[0], cv2.CONTOURS_MATCH_I3, 0)
    print(f"{angle:>7}° {s1:>12.6f} {s2:>12.6f} {s3:>12.6f}")
    results.append((angle, img, s1))

print("\n结论: Hu矩匹配对旋转具有不变性！")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('旋转不变性验证 (正方形)', fontsize=14, fontweight='bold')

for idx, (angle, img, score) in enumerate(results):
    r, c = idx // 4, idx % 4
    axes[r, c].imshow(img, cmap='gray')
    axes[r, c].set_title(f'{angle}°  I1={score:.6f}')
    axes[r, c].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_rotation_invariance.png'), dpi=150, bbox_inches='tight')
plt.show()
