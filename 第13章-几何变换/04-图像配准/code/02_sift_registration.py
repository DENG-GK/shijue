"""
示例2：SIFT特征配准
- SIFT检测器（需opencv-contrib，回退ORB）
- FLANN/BFMatcher匹配 + Lowe比率测试
- findHomography透视配准
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建纹理丰富的参考图像
np.random.seed(42)
reference = np.ones((300, 400, 3), dtype=np.uint8) * 220
for i in range(30):
    x, y = np.random.randint(20, 380), np.random.randint(20, 280)
    sz = np.random.randint(10, 30)
    color = tuple(np.random.randint(50, 200, 3).tolist())
    if np.random.rand() > 0.5:
        cv2.circle(reference, (x, y), sz, color, -1)
    else:
        cv2.rectangle(reference, (x - sz, y - sz), (x + sz, y + sz), color, -1)

h, w = reference.shape[:2]

# 透视变换
src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst_pts = np.float32([[30, 20], [w - 20, 40], [w - 40, h - 30], [20, h - 50]])
M_true = cv2.getPerspectiveTransform(src_pts, dst_pts)
source = cv2.warpPerspective(reference, M_true, (w, h), borderValue=(128, 128, 128))

gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
gray_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

# 尝试SIFT，回退ORB
try:
    detector = cv2.SIFT_create()
    norm = cv2.NORM_L2
    method_name = "SIFT"
except AttributeError:
    detector = cv2.ORB_create(nfeatures=1000)
    norm = cv2.NORM_HAMMING
    method_name = "ORB (SIFT不可用)"

kp_ref, desc_ref = detector.detectAndCompute(gray_ref, None)
kp_src, desc_src = detector.detectAndCompute(gray_src, None)
print(f"使用: {method_name}")
print(f"参考: {len(kp_ref)} 关键点, 源: {len(kp_src)} 关键点")

# 匹配
if norm == cv2.NORM_L2:
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_src, desc_ref, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
else:
    bf = cv2.BFMatcher(norm, crossCheck=True)
    good = bf.match(desc_src, desc_ref)
    good = sorted(good, key=lambda x: x.distance)[:50]

print(f"良好匹配: {len(good)}")

aligned = source.copy()
if len(good) >= 4:
    s_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    r_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_est, mask = cv2.findHomography(s_pts, r_pts, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(source, H_est, (w, h))
    inliers = mask.ravel().sum()
    print(f"内点: {inliers}/{len(good)}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f'{method_name} 特征配准', fontsize=14, fontweight='bold')

ref_kp = cv2.drawKeypoints(reference, kp_ref, None, color=(0, 255, 0))
axes[0, 0].imshow(cv2.cvtColor(ref_kp, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f'参考 ({len(kp_ref)} 关键点)')
axes[0, 0].axis('off')

src_kp = cv2.drawKeypoints(source, kp_src, None, color=(0, 255, 0))
axes[0, 1].imshow(cv2.cvtColor(src_kp, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f'源 ({len(kp_src)} 关键点)')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('对齐结果')
axes[0, 2].axis('off')

match_img = cv2.drawMatches(source, kp_src, reference, kp_ref,
                             good[:30], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
axes[1, 0].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('特征匹配')
axes[1, 0].axis('off')

overlay_before = cv2.addWeighted(reference, 0.5, source, 0.5, 0)
axes[1, 1].imshow(cv2.cvtColor(overlay_before, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('对齐前')
axes[1, 1].axis('off')

overlay_after = cv2.addWeighted(reference, 0.5, aligned, 0.5, 0)
axes[1, 2].imshow(cv2.cvtColor(overlay_after, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('对齐后')
axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_sift_registration.png'), dpi=150, bbox_inches='tight')
plt.show()
