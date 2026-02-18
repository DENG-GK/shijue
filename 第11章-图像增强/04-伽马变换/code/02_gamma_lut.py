"""
示例2：使用查找表（LUT）加速伽马变换
- LUT预计算所有256个灰度值的映射
- cv2.LUT()高效查表
- 性能对比：LUT方法远快于直接计算
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def gamma_transform_lut(image, gamma):
    """使用查找表加速伽马变换"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


# 创建测试图像
image = np.random.normal(80, 30, (500, 500)).clip(0, 255).astype(np.uint8)

# 性能对比
# 方法1：直接计算
start = time.time()
for _ in range(100):
    result1 = (np.power(image / 255.0, 2.0) * 255).astype(np.uint8)
time1 = time.time() - start

# 方法2：LUT
start = time.time()
for _ in range(100):
    result2 = gamma_transform_lut(image, 0.5)
time2 = time.time() - start

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('LUT查找表加速伽马变换', fontsize=14, fontweight='bold')

axes[0].imshow(image, cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(result1, cmap='gray')
axes[1].set_title(f'直接计算\n耗时: {time1*10:.2f}ms', fontsize=12)
axes[1].axis('off')

axes[2].imshow(result2, cmap='gray')
axes[2].set_title(f'LUT方法\n耗时: {time2*10:.2f}ms', fontsize=12)
axes[2].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '02_gamma_lut.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"直接计算: {time1*10:.2f} ms (100次)")
print(f"LUT方法:  {time2*10:.2f} ms (100次)")
print(f"LUT加速比: {time1/time2:.1f}x")
