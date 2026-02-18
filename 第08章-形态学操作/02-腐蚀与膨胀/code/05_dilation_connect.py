"""
示例5：使用膨胀连接断开的区域
这在处理断裂的字符或边缘时很有用
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建断裂的形状 =====================

def create_broken_shape():
    """创建一个断裂的形状"""
    img = np.zeros((200, 300), dtype=np.uint8)

    # 断裂的圆环
    cv2.circle(img, (80, 100), 50, 255, 8)
    cv2.rectangle(img, (75, 45), (85, 70), 0, -1)   # 打断上部
    cv2.rectangle(img, (75, 130), (85, 155), 0, -1)  # 打断下部

    # 断裂的文字模拟
    cv2.rectangle(img, (160, 60), (180, 140), 255, -1)
    cv2.rectangle(img, (180, 95), (230, 110), 255, -1)
    cv2.rectangle(img, (200, 60), (220, 140), 255, -1)
    # 添加断裂
    cv2.rectangle(img, (175, 80), (185, 90), 0, -1)
    cv2.rectangle(img, (215, 110), (225, 120), 0, -1)

    return img

# ===================== 膨胀连接 =====================

broken = create_broken_shape()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 不同迭代次数的膨胀
dilated_1 = cv2.dilate(broken, kernel, iterations=1)
dilated_2 = cv2.dilate(broken, kernel, iterations=2)
dilated_3 = cv2.dilate(broken, kernel, iterations=3)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

images = [broken, dilated_1, dilated_2, dilated_3]
titles = ['原图（断裂）', '膨胀 1 次', '膨胀 2 次', '膨胀 3 次']

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.suptitle('膨胀连接断开区域', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dilation_connect.png', dpi=150)
plt.show()

print("连接原理：")
print("• 膨胀使边缘向外扩张")
print("• 当两边扩张的区域相遇时，就连接起来了")
print("• 迭代次数决定了能连接多大的间隙")
print("• 但膨胀也会使物体整体变胖，需要后续处理")
