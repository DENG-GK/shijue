"""
示例9：轮廓检测工具类
- ContourDetector 封装
- 预处理、检测、过滤、排序
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ContourDetector:
    """轮廓检测工具类"""

    def __init__(self, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
        self.mode = mode
        self.method = method
        self.contours = []
        self.hierarchy = None

    def detect(self, img, preprocess=True, blur_size=5, threshold=127):
        """检测轮廓"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if preprocess:
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        self.contours, self.hierarchy = cv2.findContours(binary, self.mode, self.method)
        return self.contours

    def filter_by_area(self, min_area=0, max_area=float('inf')):
        """按面积过滤"""
        self.contours = [c for c in self.contours
                         if min_area <= cv2.contourArea(c) <= max_area]
        return self.contours

    def sort_contours(self, key='area', reverse=True):
        """排序轮廓"""
        if key == 'area':
            self.contours = sorted(self.contours, key=cv2.contourArea, reverse=reverse)
        elif key == 'x':
            self.contours = sorted(self.contours, key=lambda c: cv2.boundingRect(c)[0])
        elif key == 'y':
            self.contours = sorted(self.contours, key=lambda c: cv2.boundingRect(c)[1])
        return self.contours

    def get_info(self):
        """获取轮廓信息"""
        info = []
        for i, cnt in enumerate(self.contours):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y
            info.append({
                'index': i, 'area': area, 'perimeter': perimeter,
                'bbox': (x, y, w, h), 'centroid': (cx, cy), 'points': len(cnt)
            })
        return info

    def draw(self, img, color=(0, 255, 0), thickness=2, show_index=True):
        """绘制轮廓"""
        canvas = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, cnt in enumerate(self.contours):
            cv2.drawContours(canvas, [cnt], 0, color, thickness)
            if show_index:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(canvas, str(i), (cx - 5, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return canvas


# 演示
np.random.seed(42)
img = np.zeros((300, 400, 3), dtype=np.uint8)
img[:] = (240, 240, 240)
cv2.rectangle(img, (20, 20), (150, 130), (50, 50, 50), -1)
cv2.circle(img, (250, 80), 50, (50, 50, 50), -1)
cv2.ellipse(img, (340, 80), (40, 30), 0, 0, 360, (50, 50, 50), -1)
cv2.rectangle(img, (30, 170), (120, 270), (50, 50, 50), -1)
cv2.circle(img, (220, 220), 35, (50, 50, 50), -1)
for _ in range(15):
    x, y = np.random.randint(0, 400), np.random.randint(0, 300)
    cv2.circle(img, (x, y), np.random.randint(2, 6), (50, 50, 50), -1)

detector = ContourDetector()
detector.detect(img, preprocess=True, threshold=100)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('ContourDetector 工具类', fontsize=14, fontweight='bold')

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始图像')
axes[0].axis('off')

canvas1 = detector.draw(img)
axes[1].imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'全部轮廓: {len(detector.contours)}')
axes[1].axis('off')

detector.filter_by_area(min_area=500)
canvas2 = detector.draw(img)
axes[2].imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'面积>500: {len(detector.contours)}')
axes[2].axis('off')

detector.sort_contours(key='area', reverse=True)
canvas3 = detector.draw(img)
axes[3].imshow(cv2.cvtColor(canvas3, cv2.COLOR_BGR2RGB))
axes[3].set_title('按面积排序')
axes[3].axis('off')

# 打印信息
for info in detector.get_info():
    print(f"轮廓{info['index']}: 面积={info['area']:.0f}, "
          f"周长={info['perimeter']:.1f}, 点数={info['points']}")

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_contour_utils.png'), dpi=150, bbox_inches='tight')
plt.show()
