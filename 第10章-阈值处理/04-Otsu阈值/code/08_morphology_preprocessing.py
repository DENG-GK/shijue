"""
形态学预处理配合Otsu
对比直接Otsu、闭运算+Otsu、开运算+Otsu、顶帽+Otsu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 创建带噪声的测试图像 =====================

def create_noisy_image():
    img = np.zeros((300, 400), dtype=np.uint8)
    img[:, :200] = 70
    img[:, 200:] = 180

    cv2.circle(img, (100, 150), 50, 180, -1)
    cv2.rectangle(img, (250, 100), (350, 200), 70, -1)

    noise = np.random.normal(0, 25, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 形态学预处理 =====================

def morphology_then_otsu(image):
    """形态学预处理后再Otsu"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    kernel = np.ones((5, 5), np.uint8)

    # 1. 直接Otsu
    _, direct_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 闭运算（填充小洞）后Otsu
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    _, closed_otsu = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 开运算（去除噪点）后Otsu
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    _, opened_otsu = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 顶帽变换增强后Otsu
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.add(gray, tophat)
    _, tophat_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        'original': gray,
        'direct': direct_otsu,
        'closed': closed_otsu,
        'opened': opened_otsu,
        'tophat': tophat_otsu
    }

# ===================== 运行 =====================

test_img = create_noisy_image()
results = morphology_then_otsu(test_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

titles = ['原始图像', '直接Otsu', '闭运算 + Otsu',
          '开运算 + Otsu', '顶帽 + Otsu', '']
keys = ['original', 'direct', 'closed', 'opened', 'tophat', None]

for i, (title, key) in enumerate(zip(titles, keys)):
    if key is None:
        axes[i].axis('off')
    else:
        axes[i].imshow(results[key], cmap='gray')
        axes[i].set_title(title, fontsize=11)
        axes[i].axis('off')

plt.suptitle('形态学预处理配合Otsu', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('morphology_preprocessing.png', dpi=150, bbox_inches='tight')
plt.show()

print("形态学预处理的作用：")
print("- 闭运算: 填充前景中的小洞")
print("- 开运算: 去除小的噪点")
print("- 顶帽: 增强亮色细节")
