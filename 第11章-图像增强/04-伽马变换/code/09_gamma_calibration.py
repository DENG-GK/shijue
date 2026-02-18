"""
示例9：显示器伽马校正预览
- 创建伽马校准图案
- 棋盘格与实心灰度对比
- 不同显示伽马下的模拟效果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_gamma_calibration_pattern():
    """创建伽马校准图案"""
    width = 800
    height = 400

    pattern = np.zeros((height, width), dtype=np.uint8)

    # 顶部：灰度渐变
    gradient = np.tile(np.linspace(0, 255, width), (80, 1)).astype(np.uint8)
    pattern[:80, :] = gradient

    # 中间：棋盘格和对应的伽马灰度
    y_start = 100
    box_height = 60
    box_width = width // 5
    gamma_values = [1.8, 2.0, 2.2, 2.4, 2.6]

    for i, gamma in enumerate(gamma_values):
        x_start = i * box_width

        # 棋盘格
        for y in range(y_start, y_start + box_height):
            for x in range(x_start, x_start + box_width):
                if (x // 4 + y // 4) % 2 == 0:
                    pattern[y, x] = 0
                else:
                    pattern[y, x] = 255

        # 对应伽马的中灰
        target_gray = int(255 * np.power(0.5, 1 / gamma))
        solid_y = y_start + box_height + 10
        pattern[solid_y:solid_y + 40, x_start:x_start + box_width] = target_gray

    # 底部：伽马校正的渐变条
    strip_y = 260
    strip_height = 50

    for i in range(5):
        gamma = 1.8 + i * 0.2
        x_start = i * box_width
        inv_gamma = 1.0 / gamma
        for x in range(box_width):
            val = int(255 * np.power(x / box_width, inv_gamma))
            pattern[strip_y:strip_y + strip_height, x_start + x] = val

    return pattern, gamma_values


pattern, gamma_values = create_gamma_calibration_pattern()

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('显示器伽马校正预览', fontsize=14, fontweight='bold')

# 校准图案
axes[0, 0].imshow(pattern, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('校准图案（线性）', fontsize=12)
axes[0, 0].axis('off')

# 模拟不同显示器伽马
display_gammas = [1.8, 2.2, 2.6]
for i, disp_gamma in enumerate(display_gammas):
    table = np.array([((j / 255.0) ** disp_gamma) * 255
                      for j in range(256)]).astype(np.uint8)
    simulated = cv2.LUT(pattern, table)

    if i < 2:
        axes[0 + i // 2, 1].imshow(simulated, cmap='gray', vmin=0, vmax=255)
        axes[0 + i // 2, 1].set_title(f'显示器 γ = {disp_gamma}', fontsize=12)
        axes[0 + i // 2, 1].axis('off')

axes[1, 0].imshow(cv2.LUT(pattern, np.array([((j / 255.0) ** 2.2) * 255
                  for j in range(256)]).astype(np.uint8)), cmap='gray')
axes[1, 0].set_title('显示器 γ = 2.2（sRGB标准）', fontsize=12)
axes[1, 0].axis('off')

# 伽马曲线
x = np.linspace(0, 1, 100)
for gamma in [1.8, 2.0, 2.2, 2.4, 2.6]:
    axes[1, 1].plot(x, np.power(x, gamma), label=f'γ={gamma}')
axes[1, 1].set_xlabel('输入')
axes[1, 1].set_ylabel('输出')
axes[1, 1].set_title('显示器响应曲线', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '09_gamma_calibration.png'), dpi=150, bbox_inches='tight')
plt.show()

print("显示器伽马校准说明：")
print("- sRGB标准: γ = 2.2")
print("- 棋盘格应与对应灰度条亮度一致")
print("- 渐变应平滑过渡，无明显断层")
