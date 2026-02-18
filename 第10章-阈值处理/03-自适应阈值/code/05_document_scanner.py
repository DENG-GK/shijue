"""
完整的文档扫描增强流程
使用DocumentScanner类实现预处理、增强和清理
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 文档扫描增强类 =====================

class DocumentScanner:
    """文档扫描增强类"""

    def __init__(self):
        self.block_size = 21
        self.C = 10

    def preprocess(self, image):
        """预处理：转灰度、降噪"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        return denoised

    def enhance(self, image, method='gaussian'):
        """增强处理"""
        preprocessed = self.preprocess(image)

        if method == 'gaussian':
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

        binary = cv2.adaptiveThreshold(
            preprocessed, 255, adaptive_method,
            cv2.THRESH_BINARY, self.block_size, self.C
        )

        return binary

    def clean(self, binary):
        """清理：去除小噪点"""
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def process(self, image, method='gaussian', clean=True):
        """完整处理流程"""
        binary = self.enhance(image, method)
        if clean:
            binary = self.clean(binary)
        return binary

# ===================== 创建模拟扫描文档 =====================

def create_scanned_document():
    """创建模拟的扫描文档图像"""
    img = np.ones((500, 700), dtype=np.uint8) * 230

    # 不均匀背景
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            factor = 0.7 + 0.2 * np.sin(i / 100) + 0.1 * np.cos(j / 150)
            factor += 0.1 * (i / rows) - 0.05 * (j / cols)
            img[i, j] = int(img[i, j] * factor)

    # 添加标题
    cv2.putText(img, "Document Processing Demo", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 2)
    cv2.line(img, (50, 80), (550, 80), 40, 2)

    # 添加段落
    paragraphs = [
        "This is a sample scanned document with",
        "uneven lighting conditions. The adaptive",
        "thresholding method can handle this well.",
        "",
        "Key advantages:",
        "* Works with varying illumination",
        "* Preserves local details",
        "* Better text extraction quality",
        "",
        "Parameters to tune:",
        "- blockSize: neighborhood size",
        "- C: constant subtracted from mean",
    ]

    y = 120
    for para in paragraphs:
        if para:
            cv2.putText(img, para, (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 35, 1)
        y += 30

    # 添加噪声
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ===================== 使用文档扫描器 =====================

scanner = DocumentScanner()
doc_img = create_scanned_document()

result = scanner.process(doc_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(doc_img, cmap='gray')
axes[0].set_title('原始扫描文档\n(光照不均匀)', fontsize=12)
axes[0].axis('off')

axes[1].imshow(result, cmap='gray')
axes[1].set_title('增强后的文档\n(自适应阈值处理)', fontsize=12)
axes[1].axis('off')

plt.suptitle('文档扫描增强', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('document_scanner.png', dpi=150, bbox_inches='tight')
plt.show()

print("文档扫描增强完成！")
print(f"使用参数: blockSize={scanner.block_size}, C={scanner.C}")
