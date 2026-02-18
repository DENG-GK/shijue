"""
示例8：轮廓排序
- 按面积、位置、周长排序
- 从左到右/从上到下排列
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建不同大小位置的形状
img = np.zeros((400, 600), dtype=np.uint8)
shapes = [
    ('rect', (400, 50, 150, 100)),    # x=400
    ('circle', (100, 80, 40)),         # x=100
    ('rect', (250, 200, 80, 60)),      # x=250
    ('circle', (50, 300, 30)),         # x=50
    ('circle', (450, 300, 55)),        # x=450
    ('rect', (200, 30, 60, 50)),       # x=200
]

for shape_type, params in shapes:
    if shape_type == 'rect':
        x, y, w, h = params
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, -1)
    else:
        x, y, r = params
        cv2.circle(img, (x, y), r, 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 排序方式
def sort_by_area(cnts):
    return sorted(cnts, key=cv2.contourArea, reverse=True)

def sort_left_to_right(cnts):
    return sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

def sort_top_to_bottom(cnts):
    return sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

def sort_by_perimeter(cnts):
    return sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)

sort_methods = {
    '按面积 (大→小)': sort_by_area,
    '从左到右': sort_left_to_right,
    '从上到下': sort_top_to_bottom,
    '按周长 (大→小)': sort_by_perimeter,
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('轮廓排序', fontsize=14, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

# 原始编号
canvas_orig = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
for i, cnt in enumerate(contours):
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(canvas_orig, str(i), (cx - 8, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.drawContours(canvas_orig, [cnt], 0, (0, 255, 0), 2)

axes[0, 1].imshow(cv2.cvtColor(canvas_orig, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('原始顺序')
axes[0, 1].axis('off')

for idx, (name, sort_func) in enumerate(sort_methods.items()):
    sorted_cnts = sort_func(list(contours))
    canvas = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(sorted_cnts):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(canvas, str(i), (cx - 8, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)

    row = (idx + 2) // 3
    col = (idx + 2) % 3
    axes[row, col].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(name)
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_contour_sorting.png'), dpi=150, bbox_inches='tight')
plt.show()
