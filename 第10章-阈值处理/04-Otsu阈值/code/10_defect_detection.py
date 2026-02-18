"""
工业缺陷检测
使用Otsu阈值检测产品表面的缺陷（暗斑、划痕等）
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 缺陷检测函数 =====================

def detect_defects(image, min_area=50):
    """工业产品缺陷检测"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu阈值（反向，检测暗色缺陷）
    thresh, binary = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤小轮廓
    defects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            defects.append({
                'contour': contour,
                'area': area,
                'bbox': (x, y, w, h)
            })

    # 绘制结果
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for defect in defects:
        x, y, w, h = defect['bbox']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(result, f"Area:{defect['area']:.0f}", (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return binary, result, defects

# ===================== 创建模拟产品图像 =====================

def create_product_image():
    """创建带缺陷的产品表面图像"""
    img = np.ones((400, 500), dtype=np.uint8) * 180

    # 正常纹理
    for i in range(400):
        img[i, :] += (5 * np.sin(i / 20)).astype(np.uint8)

    # 缺陷
    cv2.circle(img, (100, 100), 20, 50, -1)     # 暗斑
    cv2.circle(img, (350, 150), 15, 60, -1)      # 暗斑
    cv2.line(img, (200, 50), (250, 150), 40, 3)  # 划痕
    cv2.ellipse(img, (400, 300), (25, 10), 30, 0, 360, 55, -1)  # 椭圆缺陷

    # 噪声
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 检测缺陷 =====================

product_img = create_product_image()
binary, result, defects = detect_defects(product_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(product_img, cmap='gray')
axes[0].set_title('产品表面', fontsize=12)
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('缺陷掩码 (Otsu)', fontsize=12)
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'检测结果: {len(defects)} 个缺陷', fontsize=12)
axes[2].axis('off')

plt.suptitle('工业缺陷检测', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('defect_detection.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"检测结果：发现 {len(defects)} 个缺陷")
for i, defect in enumerate(defects, 1):
    print(f"  缺陷 {i}: 面积 = {defect['area']:.0f} 像素")
