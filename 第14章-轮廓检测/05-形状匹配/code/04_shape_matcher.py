"""
示例4：形状识别系统
- ShapeMatcher 类实现多形状识别
- 模板匹配与场景检测
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ShapeMatcher:
    """形状匹配器"""

    def __init__(self, threshold=0.15):
        self.templates: Dict[str, np.ndarray] = {}
        self.threshold = threshold

    def add_template(self, name: str, contour: np.ndarray):
        self.templates[name] = contour.copy()

    def add_template_from_image(self, name: str, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            self.templates[name] = max(contours, key=cv2.contourArea)

    def match(self, contour: np.ndarray) -> List[Tuple[str, float]]:
        results = []
        for name, template in self.templates.items():
            score = cv2.matchShapes(contour, template, cv2.CONTOURS_MATCH_I1, 0)
            results.append((name, score))
        return sorted(results, key=lambda x: x[1])

    def identify(self, contour: np.ndarray) -> Tuple[str, float]:
        results = self.match(contour)
        if results and results[0][1] < self.threshold:
            return results[0]
        return ("Unknown", 1.0)


def create_template_shapes():
    templates = {}
    # 圆形
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 40, 255, -1)
    templates["circle"] = img
    # 正方形
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), 255, -1)
    templates["square"] = img
    # 三角形
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.fillPoly(img, [np.array([[50, 10], [90, 90], [10, 90]])], 255)
    templates["triangle"] = img
    # 星形
    img = np.zeros((100, 100), dtype=np.uint8)
    pts = []
    for i in range(5):
        a_out = -np.pi / 2 + i * 2 * np.pi / 5
        a_in = -np.pi / 2 + (i + 0.5) * 2 * np.pi / 5
        pts.append([int(50 + 40 * np.cos(a_out)), int(50 + 40 * np.sin(a_out))])
        pts.append([int(50 + 18 * np.cos(a_in)), int(50 + 18 * np.sin(a_in))])
    cv2.fillPoly(img, [np.array(pts)], 255)
    templates["star"] = img
    # 十字形
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (35, 10), (65, 90), 255, -1)
    cv2.rectangle(img, (10, 35), (90, 65), 255, -1)
    templates["cross"] = img
    return templates


def create_test_scene():
    img = np.ones((400, 600, 3), dtype=np.uint8) * 235
    info = []
    # 圆
    cv2.circle(img, (80, 100), 50, (80, 80, 80), -1)
    info.append(("circle", (80, 100)))
    # 旋转正方形
    pts = np.array([[-35, -35], [35, -35], [35, 35], [-35, 35]], dtype=float)
    rad = np.radians(30)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    pts = (pts @ rot.T + [220, 100]).astype(int)
    cv2.fillPoly(img, [pts], (80, 80, 80))
    info.append(("square", (220, 100)))
    # 三角形
    cv2.fillPoly(img, [np.array([[380, 40], [450, 160], [310, 160]])], (80, 80, 80))
    info.append(("triangle", (380, 100)))
    # 星形
    spts = []
    for i in range(5):
        a_o = -np.pi / 2 + i * 2 * np.pi / 5
        a_i = -np.pi / 2 + (i + 0.5) * 2 * np.pi / 5
        spts.append([int(550 + 50 * np.cos(a_o)), int(100 + 50 * np.sin(a_o))])
        spts.append([int(550 + 22 * np.cos(a_i)), int(100 + 22 * np.sin(a_i))])
    cv2.fillPoly(img, [np.array(spts)], (80, 80, 80))
    info.append(("star", (550, 100)))
    # 十字形
    cv2.rectangle(img, (60, 220), (100, 380), (80, 80, 80), -1)
    cv2.rectangle(img, (20, 270), (140, 330), (80, 80, 80), -1)
    info.append(("cross", (80, 300)))
    # 椭圆（未知）
    cv2.ellipse(img, (280, 300), (80, 50), 20, 0, 360, (80, 80, 80), -1)
    info.append(("unknown", (280, 300)))
    return img, info


if __name__ == "__main__":
    matcher = ShapeMatcher(threshold=0.2)
    templates = create_template_shapes()
    for name, img in templates.items():
        matcher.add_template_from_image(name, img)
    print("已加载模板:", list(matcher.templates.keys()))

    test_img, shapes_info = create_test_scene()
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = test_img.copy()
    sorted_cnts = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1] // 200, cv2.boundingRect(c)[0]))

    print("\n识别结果:")
    print("-" * 50)
    for i, cnt in enumerate(sorted_cnts):
        if cv2.contourArea(cnt) < 1000:
            continue
        name, score = matcher.identify(cnt)
        conf = max(0, 1 - score / 0.2) * 100
        x, y, w, h = cv2.boundingRect(cnt)
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 200)
        cv2.drawContours(canvas, [cnt], 0, color, 2)
        cv2.putText(canvas, f"{name}({conf:.0f}%)", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        print(f"  轮廓{i}: {name}, 分数={score:.4f}, 置信度={conf:.1f}%")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('形状识别系统 (ShapeMatcher)', fontsize=14, fontweight='bold')
    # 模板
    tmpl_canvas = np.ones((100, 500, 3), dtype=np.uint8) * 255
    for idx, (name, timg) in enumerate(templates.items()):
        rgb = cv2.cvtColor(timg, cv2.COLOR_GRAY2BGR)
        tmpl_canvas[:, idx * 100:(idx + 1) * 100] = rgb
    axes[0].imshow(cv2.cvtColor(tmpl_canvas, cv2.COLOR_BGR2RGB))
    axes[0].set_title('模板库')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('测试场景')
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[2].set_title('识别结果')
    axes[2].axis('off')

    plt.tight_layout()
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '04_shape_matcher.png'), dpi=150, bbox_inches='tight')
    plt.show()
