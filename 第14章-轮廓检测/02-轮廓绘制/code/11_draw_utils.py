"""
示例11：轮廓绘制工具类
- ContourDrawer 封装
- 多种绘制风格
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ContourDrawer:
    """轮廓绘制工具类"""

    def __init__(self, img_shape):
        self.h, self.w = img_shape[:2]

    def draw_contours(self, contours, color=(0, 255, 0), thickness=2):
        """基本绘制"""
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.drawContours(canvas, contours, -1, color, thickness)
        return canvas

    def draw_filled(self, contours, colors=None):
        """彩色填充"""
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i, cnt in enumerate(contours):
            if colors:
                color = colors[i % len(colors)]
            else:
                np.random.seed(i)
                color = tuple(np.random.randint(50, 255, 3).tolist())
            cv2.drawContours(canvas, [cnt], 0, color, -1)
        return canvas

    def draw_labeled(self, contours, labels=None):
        """带标签绘制"""
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i, cnt in enumerate(contours):
            cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = labels[i] if labels and i < len(labels) else str(i)
                cv2.putText(canvas, label, (cx - 10, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return canvas

    def draw_with_bboxes(self, contours, img=None):
        """带外接矩形"""
        canvas = img.copy() if img is not None else np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for cnt in contours:
            cv2.drawContours(canvas, [cnt], 0, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 1)
        return canvas

    def draw_heatmap(self, contours):
        """面积热力图"""
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        max_a = max(areas) if areas else 1
        for cnt, a in zip(contours, areas):
            ratio = a / max_a
            color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
            cv2.drawContours(canvas, [cnt], 0, color, -1)
        return canvas


# 演示
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (150, 130), 255, -1)
cv2.circle(img, (260, 80), 55, 255, -1)
cv2.ellipse(img, (120, 230), (70, 40), 0, 0, 360, 255, -1)
cv2.rectangle(img, (280, 170), (380, 280), 255, -1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
drawer = ContourDrawer(img.shape)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('ContourDrawer 工具类', fontsize=14, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

results = [
    ('基本绘制', drawer.draw_contours(contours)),
    ('彩色填充', drawer.draw_filled(contours)),
    ('带标签', drawer.draw_labeled(contours, ['矩形', '圆', '椭圆', '矩形2'])),
    ('外接矩形', drawer.draw_with_bboxes(contours)),
    ('面积热力图', drawer.draw_heatmap(contours)),
]

for idx, (name, result) in enumerate(results):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(name)
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '11_draw_utils.png'), dpi=150, bbox_inches='tight')
plt.show()
