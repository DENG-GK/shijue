"""
示例6：模板匹配与轮廓匹配结合
- 轮廓匹配筛选目标形状
- 区分匹配与不匹配对象
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_template():
    template = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(template, (40, 40), 30, 255, -1)
    return template


def create_scene():
    np.random.seed(42)
    scene = np.ones((400, 600), dtype=np.uint8) * 200
    circles = [(100, 100, 25), (300, 150, 35), (500, 100, 28)]
    for x, y, r in circles:
        cv2.circle(scene, (x, y), r, 50, -1)
    # 干扰形状
    cv2.rectangle(scene, (80, 250), (160, 350), 50, -1)
    cv2.ellipse(scene, (350, 300), (60, 40), 0, 0, 360, 50, -1)
    # 噪声
    noise = np.random.normal(0, 10, scene.shape).astype(np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return scene, circles


template = create_template()
scene, true_circles = create_scene()

# 模板轮廓
_, t_bin = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
t_cnts, _ = cv2.findContours(t_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
t_cnt = t_cnts[0]

# 场景轮廓
blurred = cv2.GaussianBlur(scene, (5, 5), 0)
_, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
matched, unmatched = 0, 0

print("轮廓匹配筛选:")
print("-" * 50)

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
    score = cv2.matchShapes(t_cnt, cnt, cv2.CONTOURS_MATCH_I1, 0)
    x, y, w, h = cv2.boundingRect(cnt)
    if score < 0.1:
        matched += 1
        cv2.drawContours(result, [cnt], 0, (0, 255, 0), 2)
        cv2.putText(result, f"Match:{score:.3f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        print(f"  轮廓{i}: 面积={area:.0f}, 分数={score:.4f} -> 匹配")
    else:
        unmatched += 1
        cv2.drawContours(result, [cnt], 0, (0, 0, 255), 2)
        cv2.putText(result, f"No:{score:.3f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        print(f"  轮廓{i}: 面积={area:.0f}, 分数={score:.4f} -> 不匹配")

print(f"\n匹配{matched}个, 不匹配{unmatched}个 (实际圆形{len(true_circles)}个)")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('模板匹配与轮廓匹配结合', fontsize=14, fontweight='bold')
axes[0].imshow(template, cmap='gray')
axes[0].set_title('模板 (圆形)')
axes[0].axis('off')
axes[1].imshow(scene, cmap='gray')
axes[1].set_title('测试场景')
axes[1].axis('off')
axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'匹配结果 (绿=匹配, 红=不匹配)')
axes[2].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_template_contour_match.png'), dpi=150, bbox_inches='tight')
plt.show()
