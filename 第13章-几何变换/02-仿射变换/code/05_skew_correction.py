"""
示例5：图像校正（倾斜校正）
- 模拟倾斜文档
- Hough直线检测倾斜角度
- getRotationMatrix2D 校正
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建模拟文档
doc = np.ones((300, 400, 3), dtype=np.uint8) * 255
for i in range(8):
    y = 40 + i * 30
    cv2.line(doc, (30, y), (370, y), (0, 0, 0), 2)
cv2.putText(doc, 'DOCUMENT', (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

h, w = doc.shape[:2]

# 人为倾斜
skew_angle = 15
skew_M = cv2.getRotationMatrix2D((w // 2, h // 2), skew_angle, 1.0)
skewed = cv2.warpAffine(doc, skew_M, (w, h), borderValue=(200, 200, 200))


def detect_skew_angle(img):
    """通过Hough变换检测倾斜角度"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return 0
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)
    return np.median(angles) if angles else 0


detected_angle = detect_skew_angle(skewed)

# 校正
correction_M = cv2.getRotationMatrix2D((w // 2, h // 2), detected_angle, 1.0)
corrected = cv2.warpAffine(skewed, correction_M, (w, h), borderValue=(255, 255, 255))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('文档倾斜校正', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(doc, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始文档')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(skewed, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f'倾斜 ({skew_angle}°)')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f'校正后 (检测: {detected_angle:.1f}°)')
axes[0, 2].axis('off')

# 边缘检测
gray = cv2.cvtColor(skewed, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
axes[1, 0].imshow(edges, cmap='gray')
axes[1, 0].set_title('Canny边缘检测')
axes[1, 0].axis('off')

# 霍夫直线
line_img = skewed.copy()
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
if lines is not None:
    for rho, theta in lines[:10, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
axes[1, 1].imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('检测到的直线')
axes[1, 1].axis('off')

axes[1, 2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_skew_correction.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"原始倾斜角度: {skew_angle}°")
print(f"检测到的角度: {detected_angle:.2f}°")
