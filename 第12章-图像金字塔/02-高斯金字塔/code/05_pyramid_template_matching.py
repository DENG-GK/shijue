"""
示例5：金字塔模板匹配
- 从粗到细加速匹配
- 与标准matchTemplate对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建场景
scene = np.ones((600, 800, 3), dtype=np.uint8) * 200
cv2.rectangle(scene, (100, 100), (180, 180), (0, 0, 255), -1)
cv2.rectangle(scene, (400, 200), (480, 280), (0, 0, 255), -1)
cv2.circle(scene, (300, 400), 60, (0, 255, 0), -1)
cv2.rectangle(scene, (600, 400), (680, 480), (0, 0, 255), -1)
noise = np.random.randint(0, 30, scene.shape, dtype=np.uint8)
scene = cv2.add(scene, noise)

# 模板
template = np.ones((80, 80, 3), dtype=np.uint8) * 200
cv2.rectangle(template, (0, 0), (80, 80), (0, 0, 255), -1)
template = cv2.add(template, np.random.randint(0, 30, template.shape, dtype=np.uint8))

# 标准匹配
result_std = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)


def pyramid_match(scene, template, levels=3):
    """金字塔从粗到细匹配"""
    scene_pyr = [scene]
    tmpl_pyr = [template]
    for i in range(levels - 1):
        scene_pyr.append(cv2.pyrDown(scene_pyr[-1]))
        tmpl_pyr.append(cv2.pyrDown(tmpl_pyr[-1]))

    candidates = []
    for level in range(levels - 1, -1, -1):
        s = scene_pyr[level]
        t = tmpl_pyr[level]
        if t.shape[0] > s.shape[0] or t.shape[1] > s.shape[1]:
            continue
        result = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.7)
        for pt in zip(*locations[::-1]):
            scale = 2 ** level
            original_pt = (pt[0] * scale, pt[1] * scale)
            score = result[pt[1], pt[0]]
            candidates.append((original_pt, score, level))
    return candidates


candidates = pyramid_match(scene, template, levels=3)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('金字塔模板匹配', fontsize=14, fontweight='bold')

scene_marked = scene.copy()
for (x, y), score, level in candidates:
    cv2.rectangle(scene_marked, (x, y), (x + template.shape[1], y + template.shape[0]),
                  (0, 255, 0), 2)
    cv2.putText(scene_marked, f'{score:.2f}', (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

axes[0, 0].imshow(cv2.cvtColor(scene_marked, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('匹配结果')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('模板')
axes[0, 1].axis('off')

axes[0, 2].imshow(result_std, cmap='hot')
axes[0, 2].set_title('标准匹配热图')
axes[0, 2].axis('off')

# 金字塔各层
scene_pyr = [scene]
for i in range(2):
    scene_pyr.append(cv2.pyrDown(scene_pyr[-1]))
for i in range(3):
    axes[1, i].imshow(cv2.cvtColor(scene_pyr[i], cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'金字塔 Level {i}')
    axes[1, i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '05_pyramid_template_matching.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n匹配结果:")
for i, ((x, y), score, level) in enumerate(candidates):
    print(f"Match {i + 1}: 位置=({x}, {y}), 分数={score:.3f}, Level={level}")
