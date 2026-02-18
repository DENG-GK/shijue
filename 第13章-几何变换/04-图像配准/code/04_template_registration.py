"""
示例4：模板匹配配准
- 多标记模板匹配定位
- estimateAffinePartial2D 估计变换
- 适用于已知标记的场景
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 参考图像（含独特标记）
reference = np.ones((300, 400, 3), dtype=np.uint8) * 200
markers = [((80, 80), (255, 0, 0)), ((320, 80), (0, 255, 0)), ((200, 220), (0, 0, 255))]
for (x, y), color in markers:
    cv2.circle(reference, (x, y), 25, color, -1)
    cv2.circle(reference, (x, y), 10, (255, 255, 255), -1)

h, w = reference.shape[:2]

# 变换后的源图像
angle, scale, tx, ty = 5, 0.95, 15, 10
M_true = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
M_true[0, 2] += tx
M_true[1, 2] += ty
source = cv2.warpAffine(reference, M_true, (w, h), borderValue=(180, 180, 180))


def find_marker(ref_img, src_img, template_center, template_size=60):
    """在源图像中通过模板匹配查找标记"""
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    x, y = template_center
    half = template_size // 2
    template = gray_ref[y - half:y + half, x - half:x + half]
    if template.shape[0] < 10 or template.shape[1] < 10:
        return None

    result = cv2.matchTemplate(gray_src, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return (max_loc[0] + half, max_loc[1] + half), max_val


# 匹配所有标记
ref_points, src_points = [], []
for (x, y), _ in markers:
    result = find_marker(reference, source, (x, y))
    if result:
        matched, conf = result
        print(f"标记({x},{y}) → 匹配到{matched} (置信度: {conf:.3f})")
        ref_points.append([x, y])
        src_points.append(list(matched))

# 估计变换
M_est, inliers = cv2.estimateAffinePartial2D(
    np.float32(src_points), np.float32(ref_points))
aligned = cv2.warpAffine(source, M_est, (w, h))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('模板匹配配准', fontsize=14, fontweight='bold')

ref_marked = reference.copy()
for (x, y), _ in markers:
    cv2.rectangle(ref_marked, (x - 30, y - 30), (x + 30, y + 30), (0, 255, 255), 2)
axes[0, 0].imshow(cv2.cvtColor(ref_marked, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('参考 (含模板)')
axes[0, 0].axis('off')

src_marked = source.copy()
for pt in src_points:
    cv2.circle(src_marked, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), 2)
axes[0, 1].imshow(cv2.cvtColor(src_marked, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('源 (匹配位置)')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('对齐结果')
axes[0, 2].axis('off')

overlay_before = cv2.addWeighted(reference, 0.5, source, 0.5, 0)
axes[1, 0].imshow(cv2.cvtColor(overlay_before, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('对齐前')
axes[1, 0].axis('off')

overlay_after = cv2.addWeighted(reference, 0.5, aligned, 0.5, 0)
axes[1, 1].imshow(cv2.cvtColor(overlay_after, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('对齐后')
axes[1, 1].axis('off')

# 点对应
for i in range(len(ref_points)):
    axes[1, 2].plot([ref_points[i][0], src_points[i][0]],
                    [ref_points[i][1], src_points[i][1]], 'b-', linewidth=2)
    axes[1, 2].plot(ref_points[i][0], ref_points[i][1], 'go', markersize=12)
    axes[1, 2].plot(src_points[i][0], src_points[i][1], 'rx', markersize=12)
axes[1, 2].invert_yaxis()
axes[1, 2].legend(['对应', '参考', '源'], fontsize=8)
axes[1, 2].set_title('点对应关系')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_template_registration.png'), dpi=150, bbox_inches='tight')
plt.show()
