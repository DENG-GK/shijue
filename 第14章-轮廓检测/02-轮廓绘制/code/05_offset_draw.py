"""
示例5：偏移绘制
- offset参数平移轮廓
- ROI区域轮廓还原到原图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建原始图像
img = np.ones((400, 600, 3), dtype=np.uint8) * 220
cv2.rectangle(img, (50, 50), (200, 200), (60, 60, 60), -1)
cv2.circle(img, (350, 130), 70, (60, 60, 60), -1)
cv2.rectangle(img, (450, 50), (550, 180), (60, 60, 60), -1)

# ROI区域
roi_x, roi_y = 30, 30
roi = img[roi_y:roi_y + 200, roi_x:roi_x + 250]
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, binary_roi = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY_INV)
contours_roi, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('偏移绘制 (offset参数)', fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# 绘制ROI框
cv2.rectangle(img, (roi_x, roi_y), (roi_x + 250, roi_y + 200), (0, 255, 0), 2)
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('原图 (绿框=ROI)')
axes[0].axis('off')

axes[1].imshow(binary_roi, cmap='gray')
axes[1].set_title('ROI二值图')
axes[1].axis('off')

# 在ROI坐标系绘制
canvas_roi = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
cv2.drawContours(canvas_roi, contours_roi, -1, (0, 255, 0), 2)
axes[2].imshow(cv2.cvtColor(canvas_roi, cv2.COLOR_BGR2RGB))
axes[2].set_title('ROI中的轮廓')
axes[2].axis('off')

# 用offset还原到原图坐标
canvas_full = img.copy()
cv2.drawContours(canvas_full, contours_roi, -1, (0, 0, 255), 2,
                 offset=(roi_x, roi_y))
axes[3].imshow(cv2.cvtColor(canvas_full, cv2.COLOR_BGR2RGB))
axes[3].set_title(f'offset=({roi_x},{roi_y}) 还原到原图')
axes[3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_offset_draw.png'), dpi=150, bbox_inches='tight')
plt.show()
