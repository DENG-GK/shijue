"""
示例7：增强现实(AR)演示
- 标记物检测模拟
- 透视变换叠加虚拟内容
- AR流水线：检测→变换→合成
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

marker_size = 200

# 创建标记
marker = np.ones((marker_size, marker_size, 3), dtype=np.uint8) * 255
cv2.rectangle(marker, (10, 10), (190, 190), (0, 0, 0), 5)
cv2.rectangle(marker, (30, 30), (70, 70), (0, 0, 0), -1)
cv2.rectangle(marker, (130, 30), (170, 70), (0, 0, 0), -1)
cv2.rectangle(marker, (30, 130), (70, 170), (0, 0, 0), -1)
cv2.rectangle(marker, (80, 80), (120, 120), (0, 0, 0), -1)

# 虚拟内容（3D线框盒子）
content = np.zeros((marker_size, marker_size, 3), dtype=np.uint8)
c = marker_size // 2
s = 60
base = np.array([[c - s, c - s], [c + s, c - s], [c + s, c + s], [c - s, c + s]], np.int32)
cv2.polylines(content, [base], True, (0, 255, 0), 2)
top = base - 20
cv2.polylines(content, [top], True, (0, 255, 0), 2)
for i in range(4):
    cv2.line(content, tuple(base[i]), tuple(top[i]), (0, 255, 0), 2)
cv2.putText(content, 'AR', (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

# 创建场景
scene = np.ones((400, 500, 3), dtype=np.uint8) * 180
noise = np.random.randint(0, 30, scene.shape, dtype=np.uint8)
scene = cv2.subtract(scene, noise)

# 标记在场景中的位置（模拟相机视角）
marker_corners = np.float32([[100, 80], [380, 50], [420, 320], [80, 350]])
marker_orig = np.float32([[0, 0], [marker_size, 0],
                           [marker_size, marker_size], [0, marker_size]])

M_marker = cv2.getPerspectiveTransform(marker_orig, marker_corners)
warped_marker = cv2.warpPerspective(marker, M_marker, (500, 400))
mask = cv2.warpPerspective(np.ones((marker_size, marker_size), dtype=np.uint8) * 255,
                            M_marker, (500, 400))

# 合成标记到场景
scene_with_marker = scene.copy()
scene_with_marker[mask > 0] = warped_marker[mask > 0]

# 叠加AR内容
warped_content = cv2.warpPerspective(content, M_marker, (500, 400))
ar_result = scene_with_marker.copy()
content_pixels = warped_content > 0
ar_result[content_pixels.any(axis=2)] = warped_content[content_pixels.any(axis=2)]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('增强现实(AR)演示', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(marker, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('标记物 (平面)')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(content, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('虚拟内容')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('空白场景')
axes[0, 2].axis('off')

scene_corners = scene_with_marker.copy()
for pt in marker_corners.astype(int):
    cv2.circle(scene_corners, tuple(pt), 8, (0, 255, 0), -1)
axes[1, 0].imshow(cv2.cvtColor(scene_corners, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('检测到标记\n(标注角点)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(ar_result, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('AR叠加结果')
axes[1, 1].axis('off')

# AR流水线
ar_text = """AR流水线:

1. 检测场景中的标记物
2. 提取标记角点
3. 计算单应矩阵:
   标记 → 场景
4. 变换虚拟内容
5. 合成到场景

应用:
- 游戏与娱乐
- 教育
- 导航
- 产品可视化
- 工业培训"""
axes[1, 2].text(0.1, 0.5, ar_text, fontsize=9, family='monospace',
                verticalalignment='center', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')
axes[1, 2].set_title('AR概念')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_ar_demo.png'), dpi=150, bbox_inches='tight')
plt.show()

print("AR演示完成！")
