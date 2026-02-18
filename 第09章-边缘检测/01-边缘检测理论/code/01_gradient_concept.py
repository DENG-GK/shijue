"""
示例1：理解图像梯度的概念
通过简单例子展示梯度与边缘的关系
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建简单的测试图像 =====================

# 创建一个有明显边缘的图像（左边暗，右边亮）
test_img = np.zeros((100, 200), dtype=np.uint8)
test_img[:, 100:] = 200  # 右半部分设为亮

print("测试图像信息：")
print(f"  尺寸: {test_img.shape}")
print(f"  左半部分像素值: {test_img[50, 50]}")
print(f"  右半部分像素值: {test_img[50, 150]}")

# ===================== 手动计算梯度 =====================

# 取中间一行来分析
row = test_img[50, :]

# 计算一阶差分（近似导数）
gradient = np.diff(row.astype(np.float64))

print(f"\n梯度分析（第50行）：")
print(f"  最大梯度值: {np.max(gradient)}")
print(f"  最大梯度位置: {np.argmax(gradient)}")
print(f"  这就是边缘的位置！")

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 原始图像
axes[0, 0].imshow(test_img, cmap='gray')
axes[0, 0].axhline(y=50, color='r', linestyle='--', label='分析的行')
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].legend()

# 灰度值分布
axes[0, 1].plot(row, 'b-', linewidth=2)
axes[0, 1].axvline(x=100, color='r', linestyle='--', label='边缘位置')
axes[0, 1].set_xlabel('像素位置')
axes[0, 1].set_ylabel('灰度值')
axes[0, 1].set_title('第50行的灰度值分布', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 梯度分布
axes[1, 0].plot(gradient, 'g-', linewidth=2)
axes[1, 0].axvline(x=99, color='r', linestyle='--', label='梯度峰值')
axes[1, 0].set_xlabel('像素位置')
axes[1, 0].set_ylabel('梯度值')
axes[1, 0].set_title('梯度（一阶差分）', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 说明文字
axes[1, 1].axis('off')
explanation = """
梯度与边缘的关系：

1. 原始图像中，左边暗（0），右边亮（200）

2. 灰度值在 x=100 处突然变化

3. 梯度（导数）在变化处出现峰值

4. 峰值位置 = 边缘位置！

结论：
边缘检测 = 找梯度大的地方
"""
axes[1, 1].text(0.1, 0.5, explanation, fontsize=12,
                verticalalignment='center', fontfamily='SimHei')

plt.tight_layout()
plt.savefig('gradient_concept.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图像已保存为 'gradient_concept.png'")
