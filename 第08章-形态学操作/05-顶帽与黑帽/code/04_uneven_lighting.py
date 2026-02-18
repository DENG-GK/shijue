"""
示例4：使用顶帽变换处理光照不均匀的图像
这是顶帽变换最重要的实际应用之一
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建光照不均匀的图像 =====================

def create_uneven_lighting_image():
    """模拟光照不均匀的文档图像"""
    img = np.zeros((200, 400), dtype=np.uint8)

    # 创建渐变背景（模拟光照不均匀）
    for x in range(400):
        brightness = int(50 + 150 * (x / 400))  # 左暗右亮
        img[:, x] = brightness

    # 添加文字（固定亮度的白色文字）
    cv2.putText(img, 'HELLO', (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                2, 255, 3)
    cv2.putText(img, 'WORLD', (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                2, 255, 3)

    # 添加一些噪声
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 处理流程 =====================

uneven = create_uneven_lighting_image()

# 1. 直接二值化（效果不好）
_, direct_binary = cv2.threshold(uneven, 150, 255, cv2.THRESH_BINARY)

# 2. 顶帽变换矫正光照
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
tophat = cv2.morphologyEx(uneven, cv2.MORPH_TOPHAT, kernel)

# 3. 对顶帽结果二值化
_, tophat_binary = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].imshow(uneven, cmap='gray')
axes[0, 0].set_title('原图（光照不均匀）\n左边暗，右边亮', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(direct_binary, cmap='gray')
axes[0, 1].set_title('直接二值化（阈值150）\n左边丢失，右边正常', fontsize=11)
axes[0, 1].axis('off')

axes[1, 0].imshow(tophat, cmap='gray')
axes[1, 0].set_title('顶帽变换后\n光照已均匀化', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(tophat_binary, cmap='gray')
axes[1, 1].set_title('顶帽后二值化\n完美提取文字！', fontsize=11)
axes[1, 1].axis('off')

plt.suptitle('顶帽变换处理光照不均匀问题', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('uneven_lighting.png', dpi=150)
plt.show()

print("处理流程：")
print("1. 使用大核（50×50）进行顶帽变换")
print("2. 大核可以估计背景光照")
print("3. 原图减去背景 = 光照均匀的前景")
print("4. 再进行简单的二值化即可得到完美结果")
