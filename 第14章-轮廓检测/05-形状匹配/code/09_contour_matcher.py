"""
示例9：轮廓匹配工具类
- ContourMatcher 封装完整匹配功能
- 模板管理、批量识别、Hu矩比较
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class MatchResult:
    """匹配结果"""
    name: str
    score: float
    confidence: float
    contour: np.ndarray


class ContourMatcher:
    """轮廓匹配工具类"""

    def __init__(self, method=cv2.CONTOURS_MATCH_I1, threshold=0.2):
        self.templates: Dict[str, np.ndarray] = {}
        self.method = method
        self.threshold = threshold

    def add_template(self, name: str, contour: np.ndarray):
        self.templates[name] = contour.copy()

    def add_template_from_image(self, name: str, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            self.templates[name] = max(cnts, key=cv2.contourArea)

    def match_all(self, contour: np.ndarray) -> List[Tuple[str, float]]:
        results = []
        for name, template in self.templates.items():
            score = cv2.matchShapes(contour, template, self.method, 0)
            results.append((name, score))
        return sorted(results, key=lambda x: x[1])

    def identify(self, contour: np.ndarray) -> MatchResult:
        results = self.match_all(contour)
        if not results:
            return MatchResult("Unknown", 1.0, 0.0, contour)
        best_name, best_score = results[0]
        conf = max(0, (1 - best_score / self.threshold)) * 100 if best_score < self.threshold else 0
        return MatchResult(
            name=best_name if best_score < self.threshold else "Unknown",
            score=best_score, confidence=conf, contour=contour
        )

    def find_in_image(self, img: np.ndarray, min_area=500) -> List[MatchResult]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [self.identify(c) for c in cnts if cv2.contourArea(c) >= min_area]

    def visualize_matches(self, img: np.ndarray, results: List[MatchResult]) -> np.ndarray:
        canvas = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for r in results:
            color = (0, 255, 0) if r.name != "Unknown" else (0, 0, 255)
            cv2.drawContours(canvas, [r.contour], 0, color, 2)
            x, y, w, h = cv2.boundingRect(r.contour)
            cv2.putText(canvas, f"{r.name}({r.confidence:.0f}%)", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return canvas

    def get_hu_moments(self, contour: np.ndarray) -> np.ndarray:
        return cv2.HuMoments(cv2.moments(contour)).flatten()

    def compare_hu(self, c1: np.ndarray, c2: np.ndarray) -> Dict:
        hu1 = -np.sign(self.get_hu_moments(c1)) * np.log10(np.abs(self.get_hu_moments(c1)) + 1e-10)
        hu2 = -np.sign(self.get_hu_moments(c2)) * np.log10(np.abs(self.get_hu_moments(c2)) + 1e-10)
        return {'hu1_log': hu1, 'hu2_log': hu2, 'diff': np.abs(hu1 - hu2),
                'score': cv2.matchShapes(c1, c2, self.method, 0)}


if __name__ == "__main__":
    matcher = ContourMatcher()

    # 添加模板
    tmpl_imgs = {}
    img = np.zeros((100, 100), np.uint8)
    cv2.circle(img, (50, 50), 40, 255, -1)
    tmpl_imgs["circle"] = img
    matcher.add_template_from_image("circle", img)

    img = np.zeros((100, 100), np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), 255, -1)
    tmpl_imgs["square"] = img
    matcher.add_template_from_image("square", img)

    img = np.zeros((100, 100), np.uint8)
    cv2.fillPoly(img, [np.array([[50, 10], [90, 90], [10, 90]])], 255)
    tmpl_imgs["triangle"] = img
    matcher.add_template_from_image("triangle", img)

    print(f"模板: {list(matcher.templates.keys())}")

    # 测试图像
    test_img = np.ones((300, 400, 3), np.uint8) * 235
    cv2.circle(test_img, (80, 100), 40, (80, 80, 80), -1)
    cv2.rectangle(test_img, (170, 60), (270, 160), (80, 80, 80), -1)
    cv2.fillPoly(test_img, [np.array([[350, 50], [390, 150], [310, 150]])], (80, 80, 80))
    cv2.ellipse(test_img, (80, 230), (50, 30), 0, 0, 360, (80, 80, 80), -1)

    results = matcher.find_in_image(test_img)
    output = matcher.visualize_matches(test_img, results)

    print("\n识别结果:")
    print("-" * 50)
    for i, r in enumerate(results):
        print(f"  {i}: {r.name}, score={r.score:.4f}, conf={r.confidence:.1f}%")

    # Hu矩比较
    if len(results) >= 2:
        comp = matcher.compare_hu(results[0].contour, results[1].contour)
        print(f"\n轮廓0 vs 轮廓1 匹配分数: {comp['score']:.4f}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('轮廓匹配工具类 (ContourMatcher)', fontsize=14, fontweight='bold')

    tmpl_bar = np.ones((100, 300, 3), dtype=np.uint8) * 255
    for idx, (name, timg) in enumerate(tmpl_imgs.items()):
        tmpl_bar[:, idx * 100:(idx + 1) * 100] = cv2.cvtColor(timg, cv2.COLOR_GRAY2BGR)
    axes[0].imshow(cv2.cvtColor(tmpl_bar, cv2.COLOR_BGR2RGB))
    axes[0].set_title('模板')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('输入图像')
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axes[2].set_title('匹配结果')
    axes[2].axis('off')

    plt.tight_layout()
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '09_contour_matcher.png'), dpi=150, bbox_inches='tight')
    plt.show()
