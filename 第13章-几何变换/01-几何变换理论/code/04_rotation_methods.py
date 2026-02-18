"""
示例4：图像旋转方法
- cv2.rotate() 90°倍数旋转
- getRotationMatrix2D + warpAffine 任意角度旋转
- 保持原尺寸 vs 保持图像完整
- 不同旋转中心的影响
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((200, 300, 3), dtype=np.uint8)
image[:, :] = [220, 220, 220]
cv2.rectangle(image, (50, 50), (250, 150), (0, 128, 255), -1)
cv2.putText(image, 'ROTATE', (70, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
cv2.circle(image, (260, 40), 20, (255, 0, 0), -1)

h, w = image.shape[:2]
center = (w // 2, h // 2)

# 方法1: cv2.rotate() — 仅支持90°倍数
rot_90_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
rot_90_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
rot_180 = cv2.rotate(image, cv2.ROTATE_180)


# 方法2: getRotationMatrix2D + warpAffine — 任意角度
def rotate_image(img, angle, scale=1.0, keep_size=True):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    if keep_size:
        rotated = cv2.warpAffine(img, M, (w, h))
    else:
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        rotated = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated


rot_30 = rotate_image(image, 30)
rot_45 = rotate_image(image, 45)
rot_30_full = rotate_image(image, 30, keep_size=False)
rot_45_scale = rotate_image(image, 45, scale=0.7)

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('图像旋转方法对比', fontsize=14, fontweight='bold')

# 原始
axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f'原始\n{w}x{h}')
axes[0, 0].axis('off')

# 90°旋转
axes[0, 1].imshow(cv2.cvtColor(rot_90_cw, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f'顺时针90°\n{rot_90_cw.shape[1]}x{rot_90_cw.shape[0]}')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(rot_90_ccw, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title(f'逆时针90°\n{rot_90_ccw.shape[1]}x{rot_90_ccw.shape[0]}')
axes[0, 2].axis('off')

axes[0, 3].imshow(cv2.cvtColor(rot_180, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(f'180°\n{rot_180.shape[1]}x{rot_180.shape[0]}')
axes[0, 3].axis('off')

# 任意角度旋转
axes[1, 0].imshow(cv2.cvtColor(rot_30, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('30° (裁切)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(rot_45, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('45° (裁切)')
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(rot_30_full, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f'30° (完整)\n{rot_30_full.shape[1]}x{rot_30_full.shape[0]}')
axes[1, 2].axis('off')

axes[1, 3].imshow(cv2.cvtColor(rot_45_scale, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title('45° + 缩放0.7')
axes[1, 3].axis('off')

# 旋转矩阵信息
M = cv2.getRotationMatrix2D(center, 30, 1.0)
matrix_text = f"getRotationMatrix2D(30°):\n\n"
matrix_text += f"[{M[0, 0]:.3f}  {M[0, 1]:.3f}  {M[0, 2]:.1f}]\n"
matrix_text += f"[{M[1, 0]:.3f}  {M[1, 1]:.3f}  {M[1, 2]:.1f}]"
axes[2, 0].text(0.1, 0.5, matrix_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=axes[2, 0].transAxes)
axes[2, 0].axis('off')
axes[2, 0].set_title('旋转矩阵')

# 不同旋转中心
M_corner = cv2.getRotationMatrix2D((0, 0), 30, 1.0)
rot_corner = cv2.warpAffine(image, M_corner, (w, h))
axes[2, 1].imshow(cv2.cvtColor(rot_corner, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title('绕(0,0)旋转30°')
axes[2, 1].axis('off')

M_br = cv2.getRotationMatrix2D((w, h), 30, 1.0)
rot_br = cv2.warpAffine(image, M_br, (w, h))
axes[2, 2].imshow(cv2.cvtColor(rot_br, cv2.COLOR_BGR2RGB))
axes[2, 2].set_title('绕右下角旋转30°')
axes[2, 2].axis('off')

axes[2, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_rotation_methods.png'), dpi=150, bbox_inches='tight')
plt.show()

print("旋转方法:")
print("  cv2.rotate(img, ROTATE_90_CLOCKWISE)    — 90°倍数")
print("  cv2.getRotationMatrix2D(center, angle, scale) + warpAffine — 任意角度")
