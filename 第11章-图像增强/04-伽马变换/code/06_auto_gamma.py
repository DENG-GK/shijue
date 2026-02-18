"""
示例6：基于图像均值的自适应伽马校正
- 根据当前均值和目标均值自动计算γ
- 公式：γ = log(target/255) / log(current_mean/255)
- 自动适应暗图像和亮图像
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def auto_gamma_correction(image, target_mean=128):
    """基于图像均值自动计算伽马值"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    current_mean = np.mean(gray)

    if 0 < current_mean < 255:
        gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
        gamma = np.clip(gamma, 0.1, 10.0)
    else:
        gamma = 1.0

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype(np.uint8)

    corrected = cv2.LUT(image, table)
    return corrected, gamma


# 创建不同曝光的测试图像
test_images = {
    '暗图像(欠曝)': np.random.randint(20, 80, (200, 300, 3), dtype=np.uint8),
    '亮图像(过曝)': np.random.randint(180, 250, (200, 300, 3), dtype=np.uint8),
    '正常图像': np.random.randint(100, 160, (200, 300, 3), dtype=np.uint8),
}

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('自适应伽马校正（基于图像均值）', fontsize=14, fontweight='bold')

for i, (name, image) in enumerate(test_images.items()):
    corrected, gamma = auto_gamma_correction(image)

    axes[0, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'原始: {name}\n均值: {np.mean(image):.1f}', fontsize=11)
    axes[0, i].axis('off')

    axes[1, i].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'校正后 (γ={gamma:.2f})\n均值: {np.mean(corrected):.1f}', fontsize=11)
    axes[1, i].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '06_auto_gamma.png'), dpi=150, bbox_inches='tight')
plt.show()

print("自适应伽马校正：")
print("- 目标均值: 128（中间灰）")
print("- 暗图像 → γ<1（自动提亮）")
print("- 亮图像 → γ>1（自动压暗）")
print("- 正常图像 → γ≈1（几乎不变）")
