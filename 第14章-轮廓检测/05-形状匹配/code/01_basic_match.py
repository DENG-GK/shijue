"""
示例1：基本形状匹配
- matchShapes 比较轮廓相似度
- 三种匹配方法对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_shape(shape_type, size=100, angle=0):
    """创建指定形状图像"""
    img = np.zeros((200, 200), dtype=np.uint8)
    cx, cy = 100, 100
    if shape_type == "circle":
        cv2.circle(img, (cx, cy), size // 2, 255, -1)
    elif shape_type == "square":
        half = size // 2
        pts = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float)
        rad = np.radians(angle)
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pts = (pts @ rot.T + [cx, cy]).astype(int)
        cv2.fillPoly(img, [pts], 255)
    elif shape_type == "triangle":
        pts = []
        for i in range(3):
            a = np.radians(angle + i * 120 - 90)
            pts.append([int(cx + size / 2 * np.cos(a)), int(cy + size / 2 * np.sin(a))])
        cv2.fillPoly(img, [np.array(pts)], 255)
    elif shape_type == "star":
        pts = []
        for i in range(5):
            a_out = np.radians(angle + i * 72 - 90)
            a_in = np.radians(angle + (i + 0.5) * 72 - 90)
            pts.append([int(cx + size / 2 * np.cos(a_out)), int(cy + size / 2 * np.sin(a_out))])
            pts.append([int(cx + size / 4 * np.cos(a_in)), int(cy + size / 4 * np.sin(a_in))])
        cv2.fillPoly(img, [np.array(pts)], 255)
    return img


def get_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0] if contours else None


# 参考形状
shapes = ["circle", "square", "triangle", "star"]
ref_imgs = {s: create_shape(s) for s in shapes}
ref_contours = {s: get_contour(img) for s, img in ref_imgs.items()}

# 测试形状：旋转30度的正方形
test_img = create_shape("square", size=80, angle=30)
test_contour = get_contour(test_img)

# 匹配
methods = [("I1", cv2.CONTOURS_MATCH_I1), ("I2", cv2.CONTOURS_MATCH_I2), ("I3", cv2.CONTOURS_MATCH_I3)]

print("形状匹配结果 (值越小越相似):")
print("-" * 60)
print(f"{'参考形状':>10} {'I1':>12} {'I2':>12} {'I3':>12}")
print("-" * 60)

scores_i1 = {}
for shape in shapes:
    row = []
    for _, method in methods:
        score = cv2.matchShapes(test_contour, ref_contours[shape], method, 0)
        row.append(score)
    scores_i1[shape] = row[0]
    print(f"{shape:>10} {row[0]:>12.4f} {row[1]:>12.4f} {row[2]:>12.4f}")

best = min(scores_i1, key=scores_i1.get)
print(f"\n最佳匹配: {best} (I1={scores_i1[best]:.4f})")

# 可视化
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
fig.suptitle('基本形状匹配 (matchShapes)', fontsize=14, fontweight='bold')

for idx, shape in enumerate(shapes):
    axes[idx].imshow(ref_imgs[shape], cmap='gray')
    label = f'{shape}\nI1={scores_i1[shape]:.4f}'
    color = 'green' if shape == best else 'black'
    axes[idx].set_title(label, color=color)
    axes[idx].axis('off')

axes[4].imshow(test_img, cmap='gray')
axes[4].set_title('测试图 (square 30°)', color='red')
axes[4].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '01_basic_match.png'), dpi=150, bbox_inches='tight')
plt.show()
