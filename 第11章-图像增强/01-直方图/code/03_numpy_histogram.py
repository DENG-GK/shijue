"""
示例3：使用NumPy计算直方图
- 对比 cv2.calcHist、np.histogram、np.bincount 三种方法
- 性能基准测试
- 验证结果一致性
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def compare_histogram_methods(image):
    """对比OpenCV和NumPy计算直方图的方法"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 方法1：OpenCV
    hist_cv = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # 方法2：NumPy histogram
    hist_np, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])

    # 方法3：NumPy bincount（最快）
    hist_bc = np.bincount(gray.flatten(), minlength=256)

    return hist_cv.flatten(), hist_np, hist_bc


# 创建测试图像
img = np.random.randint(0, 256, (500, 500), dtype=np.uint8)

# 性能对比
methods = ['cv2.calcHist', 'np.histogram', 'np.bincount']
times = []

for method in methods:
    start = time.time()
    for _ in range(100):
        if method == 'cv2.calcHist':
            cv2.calcHist([img], [0], None, [256], [0, 256])
        elif method == 'np.histogram':
            np.histogram(img.flatten(), bins=256, range=[0, 256])
        else:
            np.bincount(img.flatten(), minlength=256)
    times.append((time.time() - start) * 10)

# 验证结果一致性
hist_cv, hist_np, hist_bc = compare_histogram_methods(img)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('三种直方图计算方法对比', fontsize=14, fontweight='bold')

titles = ['cv2.calcHist', 'np.histogram', 'np.bincount']
hists = [hist_cv, hist_np, hist_bc]
colors = ['blue', 'green', 'red']

for i, (title, hist, color) in enumerate(zip(titles, hists, colors)):
    axes[i].plot(hist, color=color, linewidth=1)
    axes[i].fill_between(range(256), hist, alpha=0.3, color=color)
    axes[i].set_title(f'{title}\n耗时: {times[i]:.2f} ms', fontsize=11)
    axes[i].set_xlabel('像素值')
    axes[i].set_ylabel('频率')
    axes[i].set_xlim([0, 255])

plt.tight_layout()

# 保存图片
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_numpy_histogram.png'), dpi=150, bbox_inches='tight')
plt.show()

print("直方图计算方法对比：")
print("=" * 50)
for method, t in zip(methods, times):
    print(f"{method:15s}: {t:.2f} ms (100次平均)")
print("=" * 50)
print(f"\n结果一致性验证:")
print(f"OpenCV vs NumPy:    {np.allclose(hist_cv, hist_np)}")
print(f"OpenCV vs bincount: {np.allclose(hist_cv, hist_bc)}")
