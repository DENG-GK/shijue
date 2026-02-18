"""
示例8：手势识别应用
- 简化手势模板创建
- matchShapes 识别不同变换后的手势
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class GestureRecognizer:
    """手势识别器"""

    def __init__(self, threshold=0.2):
        self.gestures = {}
        self.threshold = threshold

    def add_gesture(self, name, template_img):
        gray = template_img if len(template_img.shape) == 2 else cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            self.gestures[name] = max(cnts, key=cv2.contourArea)

    def recognize(self, contour):
        if not self.gestures:
            return "NoTemplate", 1.0
        results = []
        for name, template in self.gestures.items():
            score = cv2.matchShapes(contour, template, cv2.CONTOURS_MATCH_I1, 0)
            results.append((name, score))
        results.sort(key=lambda x: x[1])
        if results[0][1] < self.threshold:
            return results[0]
        return "Unknown", 1.0


def create_hand_gestures():
    gestures = {}
    # 拳头
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(img, (75, 75), 50, 255, -1)
    gestures["Fist"] = img
    # 手掌
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(img, (75, 90), 35, 255, -1)
    for i in range(5):
        angle = -np.pi / 2 + (i - 2) * np.pi / 8
        x1 = int(75 + 30 * np.cos(angle))
        y1 = int(90 + 30 * np.sin(angle))
        x2 = int(75 + 65 * np.cos(angle))
        y2 = int(90 + 65 * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), 255, 10)
    gestures["Open"] = img
    # V形
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(img, (75, 100), 25, 255, -1)
    cv2.line(img, (60, 90), (50, 30), 255, 10)
    cv2.line(img, (90, 90), (100, 30), 255, 10)
    gestures["Victory"] = img
    # 点赞
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.ellipse(img, (75, 100), (30, 25), 0, 0, 360, 255, -1)
    cv2.line(img, (75, 85), (75, 30), 255, 15)
    gestures["ThumbUp"] = img
    return gestures


recognizer = GestureRecognizer()
gesture_templates = create_hand_gestures()
for name, img in gesture_templates.items():
    recognizer.add_gesture(name, img)

# 测试
test_cases = [
    ("Fist", 1.0, 0), ("Fist", 0.8, 30),
    ("Open", 1.0, 0), ("Open", 1.2, -15),
    ("Victory", 1.0, 0), ("Victory", 0.9, 45),
    ("ThumbUp", 1.0, 0), ("ThumbUp", 1.1, 20),
]

print("手势识别测试:")
print("-" * 65)
print(f"{'实际':>10} {'变换':>20} {'识别':>10} {'分数':>10} {'正确':>5}")
print("-" * 65)

results = []
for gesture, scale, rotation in test_cases:
    template = gesture_templates[gesture]
    h, w = template.shape
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(template, (new_w, new_h))
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), rotation, 1.0)
    rotated = cv2.warpAffine(scaled, M, (new_w, new_h))

    cnts, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        recognized, score = recognizer.recognize(largest)
        ok = "Y" if recognized == gesture else "N"
        print(f"{gesture:>10} s={scale},r={rotation:>3}° {recognized:>10} {score:>10.4f} {ok:>5}")
        results.append((gesture, rotated, recognized, score))

# 可视化
fig = plt.figure(figsize=(16, 10))
fig.suptitle('手势识别应用', fontsize=14, fontweight='bold')

# 模板行
for idx, (name, img) in enumerate(gesture_templates.items()):
    ax = fig.add_subplot(3, 4, idx + 1)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'模板: {name}')
    ax.axis('off')

# 结果行
for idx, (actual, img, recognized, score) in enumerate(results):
    ax = fig.add_subplot(3, 4, 5 + idx)
    ax.imshow(img, cmap='gray')
    color = 'green' if recognized == actual else 'red'
    ax.set_title(f'{recognized}({score:.3f})', color=color, fontsize=9)
    ax.axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '08_gesture_recognition.png'), dpi=150, bbox_inches='tight')
plt.show()
