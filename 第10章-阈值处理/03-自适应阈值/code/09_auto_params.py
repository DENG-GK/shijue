"""
自动参数选择策略
根据图像特征自动选择blockSize和C参数
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 自动参数选择函数 =====================

def auto_adaptive_params(image):
    """根据图像特征自动选择自适应阈值参数"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape

    # 1. 估算blockSize：基于图像尺寸
    min_dim = min(height, width)
    suggested_block = max(11, min(min_dim // 15, 51))
    suggested_block = suggested_block if suggested_block % 2 == 1 else suggested_block + 1

    # 2. 估算C值：基于图像对比度
    std_dev = np.std(gray)
    suggested_c = max(2, min(int(std_dev / 10), 15))

    # 3. 验证参数（通过分析结果质量）
    results = {}
    for block in [suggested_block - 10, suggested_block, suggested_block + 10]:
        if block < 3:
            continue
        block = block if block % 2 == 1 else block + 1

        for c in [suggested_c - 3, suggested_c, suggested_c + 3]:
            if c < 0:
                continue

            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block, c)

            white_ratio = np.sum(binary == 255) / binary.size

            if 0.2 < white_ratio < 0.8:
                quality_score = 1 - abs(white_ratio - 0.5)
            else:
                quality_score = 0

            results[(block, c)] = {
                'binary': binary,
                'white_ratio': white_ratio,
                'score': quality_score
            }

    # 选择最佳参数
    best_params = max(results.keys(), key=lambda x: results[x]['score'])

    return {
        'block_size': best_params[0],
        'C': best_params[1],
        'result': results[best_params]['binary'],
        'analysis': {
            'image_size': (height, width),
            'std_dev': std_dev,
            'white_ratio': results[best_params]['white_ratio']
        }
    }

# ===================== 创建测试图像 =====================

def create_test_image_for_auto():
    img = np.ones((300, 400), dtype=np.uint8) * 200
    cv2.putText(img, "Auto Params Test", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 40, 2)
    cv2.putText(img, "Testing automatic", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 50, 2)
    cv2.putText(img, "parameter selection", (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 50, 2)

    for j in range(400):
        factor = 0.6 + 0.4 * (j / 400)
        img[:, j] = (img[:, j] * factor).astype(np.uint8)

    noise = np.random.normal(0, 10, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

# ===================== 运行自动参数选择 =====================

test_img = create_test_image_for_auto()
auto_result = auto_adaptive_params(test_img)

# ===================== 可视化 =====================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(test_img, cmap='gray')
axes[0].set_title('原始图像', fontsize=12)
axes[0].axis('off')

axes[1].imshow(auto_result['result'], cmap='gray')
axes[1].set_title(f"自动参数\nblockSize={auto_result['block_size']}, C={auto_result['C']}", fontsize=12)
axes[1].axis('off')

plt.suptitle('自动参数选择', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('auto_params.png', dpi=150, bbox_inches='tight')
plt.show()

print("自动参数选择结果：")
print(f"  blockSize: {auto_result['block_size']}")
print(f"  C: {auto_result['C']}")
print(f"\n图像分析：")
print(f"  尺寸: {auto_result['analysis']['image_size']}")
print(f"  标准差: {auto_result['analysis']['std_dev']:.2f}")
print(f"  白色比例: {auto_result['analysis']['white_ratio']:.2%}")
