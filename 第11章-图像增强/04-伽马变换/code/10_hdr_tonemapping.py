"""
示例10：HDR图像色调映射
- 创建模拟HDR图像（高动态范围）
- Reinhard色调映射 + 伽马校正
- 对比不同曝光和伽马参数的效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def gamma_based_tone_mapping(hdr_image, gamma=2.2, exposure=1.0):
    """基于伽马的简单色调映射"""
    exposed = hdr_image * exposure
    exposed = np.maximum(exposed, 0)

    # Reinhard色调映射
    mapped = exposed / (1 + exposed)

    # 伽马校正
    inv_gamma = 1.0 / gamma
    corrected = np.power(mapped, inv_gamma)

    ldr = np.clip(corrected * 255, 0, 255).astype(np.uint8)
    return ldr


def create_synthetic_hdr():
    """创建合成HDR图像"""
    height, width = 300, 400
    hdr = np.zeros((height, width, 3), dtype=np.float32)

    # 亮天空
    hdr[:height // 2, :, 0] = 0.8
    hdr[:height // 2, :, 1] = 0.9
    hdr[:height // 2, :, 2] = 1.2

    # 暗地面
    hdr[height // 2:, :, 0] = 0.1
    hdr[height // 2:, :, 1] = 0.15
    hdr[height // 2:, :, 2] = 0.05

    # 太阳（非常亮）
    cv2.circle(hdr, (width // 4, height // 3), 30, (50, 40, 30), -1)
    cv2.circle(hdr, (width // 4, height // 3), 15, (100, 80, 60), -1)

    noise = np.random.normal(0, 0.02, hdr.shape).astype(np.float32)
    hdr = hdr + noise

    return hdr


hdr_image = create_synthetic_hdr()

# 不同参数的色调映射
exposures = [0.5, 1.0, 2.0]
gammas = [1.8, 2.2, 2.6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('HDR色调映射 - 伽马与曝光', fontsize=14, fontweight='bold')

# 不同曝光
for i, exp in enumerate(exposures):
    result = gamma_based_tone_mapping(hdr_image, gamma=2.2, exposure=exp)
    axes[0, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'曝光={exp}, γ=2.2', fontsize=11)
    axes[0, i].axis('off')

# 不同伽马
for i, gam in enumerate(gammas):
    result = gamma_based_tone_mapping(hdr_image, gamma=gam, exposure=1.0)
    axes[1, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'曝光=1.0, γ={gam}', fontsize=11)
    axes[1, i].axis('off')

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '10_hdr_tonemapping.png'), dpi=150, bbox_inches='tight')
plt.show()

print("HDR色调映射说明：")
print("- 曝光调整：控制整体亮度")
print("- 伽马校正：调整显示特性")
print("- Reinhard映射：将HDR值压缩到LDR范围")
