"""
示例5：道路鸟瞰图变换
- 模拟道路透视场景
- 梯形→矩形变换
- 车道线在BEV中变为平行线
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建模拟道路场景
image = np.zeros((400, 600, 3), dtype=np.uint8)
image[:150, :] = [200, 180, 150]  # 天空

# 道路（梯形，近大远小）
road_pts = np.array([[200, 150], [400, 150], [600, 400], [0, 400]], np.int32)
cv2.fillPoly(image, [road_pts], (80, 80, 80))

# 车道标线
for i in range(3):
    for offset in [-100, 100]:
        y_start = 160 + i * 80
        y_end = y_start + 40
        x_top = 300 + offset * (1 - (y_start - 150) / 250 * 0.6)
        x_bottom = 300 + offset * (1 - (y_end - 150) / 250 * 0.6)
        pts = np.array([
            [x_top - 5, y_start], [x_top + 5, y_start],
            [x_bottom + 8, y_end], [x_bottom - 8, y_end]
        ], np.int32)
        cv2.fillPoly(image, [pts], (255, 255, 255))

# 中线
for i in range(4):
    y_s = 155 + i * 60
    y_e = y_s + 30
    pts = np.array([[298, y_s], [302, y_s], [302, y_e], [298, y_e]], np.int32)
    cv2.fillPoly(image, [pts], (255, 255, 0))

h, w = image.shape[:2]

# 透视变换点
src_pts = np.float32([[200, 150], [400, 150], [550, 380], [50, 380]])
dst_pts = np.float32([[150, 0], [450, 0], [450, 400], [150, 400]])

M_bev = cv2.getPerspectiveTransform(src_pts, dst_pts)
bev = cv2.warpPerspective(image, M_bev, (w, h))

M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
recovered = cv2.warpPerspective(bev, M_inv, (w, h))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('道路鸟瞰图(BEV)变换', fontsize=14, fontweight='bold')

# 标记ROI
img_marked = image.copy()
cv2.polylines(img_marked, [src_pts.astype(int)], True, (0, 255, 0), 2)
for pt in src_pts.astype(int):
    cv2.circle(img_marked, tuple(pt), 8, (0, 255, 0), -1)
axes[0, 0].imshow(cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始视角\n(标记ROI)')
axes[0, 0].axis('off')

bev_marked = bev.copy()
cv2.polylines(bev_marked, [dst_pts.astype(int)], True, (0, 255, 0), 2)
axes[0, 1].imshow(cv2.cvtColor(bev_marked, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('鸟瞰视角 (BEV)')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(recovered, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('恢复视角')
axes[0, 2].axis('off')

# 网格对比
grid_img = image.copy()
for y in range(150, 400, 30):
    cv2.line(grid_img, (0, y), (600, y), (0, 100, 0), 1)
for x in range(0, 600, 40):
    cv2.line(grid_img, (x, 150), (x, 400), (0, 100, 0), 1)
axes[1, 0].imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('原始+网格')
axes[1, 0].axis('off')

grid_bev = cv2.warpPerspective(grid_img, M_bev, (w, h))
axes[1, 1].imshow(cv2.cvtColor(grid_bev, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('BEV+网格\n(平行线恢复)')
axes[1, 1].axis('off')

# 应用说明
apps_text = """鸟瞰图应用:

1. 车道线检测
   - 平行车道更易检测
   - 曲率计算

2. 距离测量
   - 精确距离估计
   - 物体间距

3. 路径规划
   - 自动驾驶
   - 泊车辅助

关键优势:
- 消除透视畸变
- 使平行线真正平行
- 允许度量测量"""
axes[1, 2].text(0.05, 0.5, apps_text, fontsize=9, family='monospace',
                verticalalignment='center', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')
axes[1, 2].set_title('应用场景')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_birds_eye_view.png'), dpi=150, bbox_inches='tight')
plt.show()

print("鸟瞰图变换完成！")
