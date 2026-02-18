"""
示例10：轮廓平滑
- 高斯平滑、移动平均、多边形近似
- 不同平滑方法对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def smooth_contour_gaussian(contour, sigma=3):
    """高斯滤波平滑轮廓"""
    x = contour[:, 0, 0].astype(float)
    y = contour[:, 0, 1].astype(float)
    n = len(x)

    # 手动实现高斯平滑（循环边界），避免scipy依赖
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()

    # 循环扩展
    x_ext = np.concatenate([x[-half:], x, x[:half]])
    y_ext = np.concatenate([y[-half:], y, y[:half]])
    x_smooth = np.convolve(x_ext, kernel, mode='valid')
    y_smooth = np.convolve(y_ext, kernel, mode='valid')

    smoothed = np.zeros_like(contour)
    smoothed[:, 0, 0] = np.round(x_smooth).astype(int)
    smoothed[:, 0, 1] = np.round(y_smooth).astype(int)
    return smoothed


def smooth_contour_moving_average(contour, window=5):
    """移动平均平滑轮廓"""
    x = contour[:, 0, 0].astype(float)
    y = contour[:, 0, 1].astype(float)
    n = len(x)

    kernel = np.ones(window) / window
    # 循环扩展防止边界效应
    x_ext = np.concatenate([x[-(window//2):], x, x[:window//2]])
    y_ext = np.concatenate([y[-(window//2):], y, y[:window//2]])
    x_smooth = np.convolve(x_ext, kernel, mode='valid')
    y_smooth = np.convolve(y_ext, kernel, mode='valid')

    # 截取到原始长度
    x_smooth = x_smooth[:n]
    y_smooth = y_smooth[:n]

    smoothed = np.zeros_like(contour)
    smoothed[:, 0, 0] = np.round(x_smooth).astype(int)
    smoothed[:, 0, 1] = np.round(y_smooth).astype(int)
    return smoothed


def smooth_contour_approx(contour, epsilon_ratio=0.005):
    """多边形近似平滑"""
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * perimeter
    return cv2.approxPolyDP(contour, epsilon, True)


# 创建带噪声的轮廓
np.random.seed(42)
img = np.zeros((400, 400), dtype=np.uint8)
center = (200, 200)
radius = 120
pts = []
for i in range(100):
    angle = i * 2 * np.pi / 100
    noise = np.random.randint(-10, 10)
    r = radius + noise
    x = int(center[0] + r * np.cos(angle))
    y = int(center[1] + r * np.sin(angle))
    pts.append([x, y])
cv2.fillPoly(img, [np.array(pts)], 255)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

# 应用不同平滑方法
methods = {
    "原始轮廓": cnt,
    "Gaussian σ=3": smooth_contour_gaussian(cnt, 3),
    "Gaussian σ=5": smooth_contour_gaussian(cnt, 5),
    "Moving Avg w=5": smooth_contour_moving_average(cnt, 5),
    "Moving Avg w=10": smooth_contour_moving_average(cnt, 10),
    "Approx ε=0.005": smooth_contour_approx(cnt, 0.005),
}

original_perimeter = cv2.arcLength(cnt, True)

print("轮廓平滑方法比较:")
print("-" * 55)
print(f"{'方法':>20} {'点数':>8} {'周长变化':>12}")
print("-" * 55)

for name, smoothed in methods.items():
    p = cv2.arcLength(smoothed, True)
    change = (p - original_perimeter) / original_perimeter * 100
    print(f"{name:>20} {len(smoothed):>8} {change:>+11.1f}%")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('轮廓平滑方法比较', fontsize=14, fontweight='bold')

for idx, (name, smoothed) in enumerate(methods.items()):
    row, col = idx // 3, idx % 3
    canvas = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_GRAY2BGR)
    # 原始轮廓（灰色）
    cv2.drawContours(canvas, [cnt], 0, (60, 60, 60), 1)
    # 平滑轮廓（绿色）
    cv2.drawContours(canvas, [smoothed], 0, (0, 255, 0), 2)

    p = cv2.arcLength(smoothed, True)
    change = (p - original_perimeter) / original_perimeter * 100

    axes[row, col].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f'{name}\n{len(smoothed)}点, 周长{change:+.1f}%')
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_contour_smooth.png'), dpi=150, bbox_inches='tight')
plt.show()
