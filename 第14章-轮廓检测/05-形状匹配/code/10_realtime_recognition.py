"""
示例10：实时形状识别
- RealTimeShapeRecognizer 模拟实时场景
- 随机场景生成与批量识别
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class RealTimeShapeRecognizer:
    """实时形状识别器"""

    def __init__(self, threshold=0.15, min_area=1000):
        self.templates = {}
        self.threshold = threshold
        self.min_area = min_area

    def load_templates(self):
        # 圆
        img = np.zeros((100, 100), np.uint8)
        cv2.circle(img, (50, 50), 40, 255, -1)
        self._add("circle", img)
        # 正方形
        img = np.zeros((100, 100), np.uint8)
        cv2.rectangle(img, (10, 10), (90, 90), 255, -1)
        self._add("square", img)
        # 三角形
        img = np.zeros((100, 100), np.uint8)
        cv2.fillPoly(img, [np.array([[50, 10], [90, 90], [10, 90]])], 255)
        self._add("triangle", img)
        # 星形
        img = np.zeros((100, 100), np.uint8)
        pts = []
        for i in range(5):
            a_o = -np.pi / 2 + i * 2 * np.pi / 5
            a_i = -np.pi / 2 + (i + 0.5) * 2 * np.pi / 5
            pts.append([int(50 + 40 * np.cos(a_o)), int(50 + 40 * np.sin(a_o))])
            pts.append([int(50 + 18 * np.cos(a_i)), int(50 + 18 * np.sin(a_i))])
        cv2.fillPoly(img, [np.array(pts)], 255)
        self._add("star", img)

    def _add(self, name, img):
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            self.templates[name] = cnts[0]

    def recognize(self, contour):
        best_name, best_score = "Unknown", float('inf')
        for name, tmpl in self.templates.items():
            score = cv2.matchShapes(contour, tmpl, cv2.CONTOURS_MATCH_I1, 0)
            if score < best_score:
                best_score = score
                best_name = name
        if best_score > self.threshold:
            best_name = "Unknown"
        return best_name, best_score

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            name, score = self.recognize(cnt)
            results.append((cnt, name, score))
        return results, binary

    def draw_results(self, frame, results):
        output = frame.copy()
        colors = {'circle': (0, 200, 0), 'square': (200, 0, 0), 'triangle': (0, 0, 200),
                  'star': (200, 200, 0), 'Unknown': (128, 128, 128)}
        for cnt, name, score in results:
            color = colors.get(name, (128, 128, 128))
            cv2.drawContours(output, [cnt], 0, color, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            conf = max(0, (1 - score / self.threshold) * 100)
            cv2.putText(output, f"{name}({conf:.0f}%)", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        return output


def create_random_scene(seed=None):
    if seed is not None:
        np.random.seed(seed)
    frame = np.ones((400, 600, 3), np.uint8) * 235
    shape_types = ['circle', 'square', 'triangle', 'star']
    n = np.random.randint(4, 8)
    placed = []
    for _ in range(n):
        shape = np.random.choice(shape_types)
        x = np.random.randint(80, 520)
        y = np.random.randint(80, 320)
        size = np.random.randint(30, 55)
        angle = np.random.randint(0, 360)
        if shape == 'circle':
            cv2.circle(frame, (x, y), size, (80, 80, 80), -1)
        elif shape == 'square':
            pts = np.array([[-size, -size], [size, -size], [size, size], [-size, size]], dtype=float)
            rad = np.radians(angle)
            rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            pts = (pts @ rot.T + [x, y]).astype(int)
            cv2.fillPoly(frame, [pts], (80, 80, 80))
        elif shape == 'triangle':
            pts = []
            for i in range(3):
                a = np.radians(angle + i * 120)
                pts.append([int(x + size * np.cos(a)), int(y + size * np.sin(a))])
            cv2.fillPoly(frame, [np.array(pts)], (80, 80, 80))
        elif shape == 'star':
            pts = []
            for i in range(5):
                a_o = np.radians(angle + i * 72)
                a_i = np.radians(angle + (i + 0.5) * 72)
                pts.append([int(x + size * np.cos(a_o)), int(y + size * np.sin(a_o))])
                pts.append([int(x + size * 0.4 * np.cos(a_i)), int(y + size * 0.4 * np.sin(a_i))])
            cv2.fillPoly(frame, [np.array(pts)], (80, 80, 80))
        placed.append(shape)
    return frame, placed


if __name__ == "__main__":
    recognizer = RealTimeShapeRecognizer()
    recognizer.load_templates()

    # 生成多个场景
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('实时形状识别 - 多场景演示', fontsize=14, fontweight='bold')

    for idx in range(6):
        frame, placed = create_random_scene(seed=idx * 7 + 42)
        t0 = time.time()
        results, binary = recognizer.process_frame(frame)
        elapsed = (time.time() - t0) * 1000
        output = recognizer.draw_results(frame, results)

        row, col = idx // 3, idx % 3
        axes[row, col].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        detected = [name for _, name, _ in results if name != "Unknown"]
        axes[row, col].set_title(f'场景{idx + 1}: 检测{len(detected)}个 ({elapsed:.1f}ms)')
        axes[row, col].axis('off')

        print(f"场景{idx + 1}: 放置{len(placed)}个, 检测{len(results)}个, "
              f"识别{len(detected)}个, 耗时{elapsed:.1f}ms")

    plt.tight_layout()
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '10_realtime_recognition.png'), dpi=150, bbox_inches='tight')
    plt.show()
