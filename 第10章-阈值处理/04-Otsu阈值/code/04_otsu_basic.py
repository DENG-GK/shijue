"""
OpenCV中Otsu阈值的基本使用
演示标准用法和参数说明
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== Otsu基本用法 =====================

def demo_otsu_usage():
    """演示Otsu阈值的基本使用"""

    # 创建测试图像
    img = np.zeros((200, 300), dtype=np.uint8)
    img[:, :150] = 50   # 左半部分暗
    img[:, 150:] = 200  # 右半部分亮

    # 添加噪声
    noise = np.random.normal(0, 20, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # 使用Otsu阈值
    # 注意：thresh参数设为0，Otsu会自动计算最佳阈值
    otsu_thresh, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("=" * 50)
    print("Otsu阈值使用方法：")
    print("=" * 50)
    print("代码：")
    print("  ret, binary = cv2.threshold(gray, 0, 255,")
    print("                cv2.THRESH_BINARY + cv2.THRESH_OTSU)")
    print()
    print("参数说明：")
    print("  - thresh=0: 传入的阈值会被忽略")
    print("  - 返回的ret: 实际使用的Otsu阈值")
    print("  - THRESH_OTSU: 启用Otsu自动阈值")
    print()
    print(f"本例中Otsu自动选择的阈值: {otsu_thresh:.0f}")
    print("=" * 50)

    return img, binary, otsu_thresh

# ===================== 运行 =====================

img, binary, thresh = demo_otsu_usage()

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title(f'Otsu (T={thresh:.0f})', fontsize=12)
axes[1].axis('off')

plt.suptitle('Otsu阈值基本使用', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('otsu_basic.png', dpi=150, bbox_inches='tight')
plt.show()
