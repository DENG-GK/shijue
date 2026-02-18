"""
示例3：手动构建各类仿射变换矩阵
- 平移/旋转/缩放/错切/翻转矩阵
- 显示矩阵数值与对应效果
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
cv2.rectangle(image, (50, 30), (250, 170), (0, 128, 255), -1)
cv2.circle(image, (80, 60), 15, (255, 0, 0), -1)
cv2.putText(image, 'M', (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

h, w = image.shape[:2]
cx, cy = w // 2, h // 2


def create_translation_matrix(tx, ty):
    return np.float32([[1, 0, tx], [0, 1, ty]])


def create_rotation_matrix(angle, center):
    return cv2.getRotationMatrix2D(center, angle, 1.0)


def create_scale_matrix(sx, sy, center):
    return np.float32([
        [sx, 0, center[0] * (1 - sx)],
        [0, sy, center[1] * (1 - sy)]
    ])


def create_shear_matrix(shx, shy):
    return np.float32([[1, shx, 0], [shy, 1, 0]])


def create_flip_matrix(flip_x, flip_y, center):
    sx = -1 if flip_x else 1
    sy = -1 if flip_y else 1
    return np.float32([
        [sx, 0, center[0] * (1 - sx)],
        [0, sy, center[1] * (1 - sy)]
    ])


transformations = {
    '原始': np.float32([[1, 0, 0], [0, 1, 0]]),
    '平移 (50,20)': create_translation_matrix(50, 20),
    '旋转 30°': create_rotation_matrix(30, (cx, cy)),
    '缩放 (1.3,0.7)': create_scale_matrix(1.3, 0.7, (cx, cy)),
    '错切 X(0.3)': create_shear_matrix(0.3, 0),
    '错切 Y(0.3)': create_shear_matrix(0, 0.3),
    '水平翻转': create_flip_matrix(True, False, (cx, cy)),
    '垂直翻转': create_flip_matrix(False, True, (cx, cy)),
}

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('手动构建仿射变换矩阵', fontsize=14, fontweight='bold')

for i, (name, M) in enumerate(transformations.items()):
    row, col = i // 4, i % 4
    result = cv2.warpAffine(image, M, (w, h))

    axes[row, col].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    title = f'{name}\n[{M[0, 0]:.2f} {M[0, 1]:.2f} {M[0, 2]:.1f}]'
    title += f'\n[{M[1, 0]:.2f} {M[1, 1]:.2f} {M[1, 2]:.1f}]'
    axes[row, col].set_title(title, fontsize=9)
    axes[row, col].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '03_build_affine_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()

print("手动构建仿射变换矩阵完成！")
