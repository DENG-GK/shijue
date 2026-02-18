"""
示例3：文档透视校正
- 模拟拍照文档（透视畸变）
- 四角定义→矩形校正
- 校正前后对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建"文档"
doc = np.ones((400, 300, 3), dtype=np.uint8) * 255
cv2.rectangle(doc, (20, 20), (280, 380), (0, 0, 0), 2)
for i in range(15):
    y = 50 + i * 22
    cv2.line(doc, (40, y), (260, y), (100, 100, 100), 1)
cv2.putText(doc, 'DOCUMENT', (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(doc, 'Page 1', (110, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

h, w = doc.shape[:2]

# 透视畸变（模拟拍照）
src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst_pts = np.float32([[50, 80], [380, 30], [420, 380], [30, 420]])

canvas_size = (450, 500)
canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 180
noise = np.random.randint(0, 20, canvas.shape, dtype=np.uint8)
canvas = cv2.add(canvas, noise)

M_distort = cv2.getPerspectiveTransform(src_pts, dst_pts)
distorted_doc = cv2.warpPerspective(doc, M_distort, canvas_size)
mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, M_distort, canvas_size)
canvas[mask > 0] = distorted_doc[mask > 0]

# 校正
detected_corners = dst_pts
target_h, target_w = 400, 300
target_pts = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
M_correct = cv2.getPerspectiveTransform(detected_corners, target_pts)
corrected = cv2.warpPerspective(canvas, M_correct, (target_w, target_h))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('文档透视校正', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(doc, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始文档')
axes[0, 0].axis('off')

canvas_marked = canvas.copy()
for i, pt in enumerate(detected_corners.astype(int)):
    cv2.circle(canvas_marked, tuple(pt), 8, (0, 255, 0), -1)
    cv2.putText(canvas_marked, str(i + 1), (pt[0] + 10, pt[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
axes[0, 1].imshow(cv2.cvtColor(canvas_marked, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('拍照文档\n(标记角点)')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('校正后文档')
axes[0, 2].axis('off')

# 叠加对比
doc_resized = cv2.resize(doc, (target_w, target_h))
overlay = cv2.addWeighted(doc_resized, 0.5, corrected, 0.5, 0)
axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('原始 vs 校正 (叠加)')
axes[1, 0].axis('off')

# 角点映射
axes[1, 1].set_title('角点映射')
for i in range(4):
    s = detected_corners[i]
    d = target_pts[i]
    axes[1, 1].plot([s[0], d[0] + 500], [s[1], d[1]], 'b-', linewidth=1.5)
    axes[1, 1].plot(s[0], s[1], 'ro', markersize=10)
    axes[1, 1].plot(d[0] + 500, d[1], 'go', markersize=10)
axes[1, 1].axvline(x=450, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].text(200, 470, '畸变', ha='center', fontsize=12)
axes[1, 1].text(650, 470, '校正', ha='center', fontsize=12)
axes[1, 1].set_xlim(-50, 850)
axes[1, 1].set_ylim(500, -50)

axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_document_correction.png'), dpi=150, bbox_inches='tight')
plt.show()

print("文档透视校正完成！")
