"""
示例4：组合多个仿射变换
- 逐步变换可视化：居中→缩放→旋转→平移
- 矩阵乘法组合 M = T2 @ R @ S @ T1
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((300, 400, 3), dtype=np.uint8)
image[:, :] = [200, 200, 200]
cv2.rectangle(image, (100, 75), (300, 225), (0, 150, 255), -1)
cv2.circle(image, (130, 105), 15, (255, 0, 0), -1)
cv2.putText(image, 'COMBO', (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

h, w = image.shape[:2]
cx, cy = w // 2, h // 2


def to_3x3(M):
    return np.vstack([M, [0, 0, 1]])


def to_2x3(M):
    return M[:2, :].astype(np.float32)


# 步骤1: 平移到原点
T1 = to_3x3(np.float32([[1, 0, -cx], [0, 1, -cy]]))

# 步骤2: 缩放
S = to_3x3(np.float32([[1.2, 0, 0], [0, 0.8, 0]]))

# 步骤3: 旋转
angle = np.radians(25)
R = to_3x3(np.float32([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0]
]))

# 步骤4: 平移回+偏移
T2 = to_3x3(np.float32([[1, 0, cx + 30], [0, 1, cy + 20]]))

# 组合: T2 @ R @ S @ T1
M_combined = T2 @ R @ S @ T1

steps = {
    '原始': np.float32([[1, 0, 0], [0, 1, 0]]),
    '步骤1: 居中': to_2x3(T1),
    '步骤2: +缩放': to_2x3(S @ T1),
    '步骤3: +旋转': to_2x3(R @ S @ T1),
    '步骤4: +平移': to_2x3(M_combined),
}

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle('仿射变换逐步组合', fontsize=14, fontweight='bold')

for i, (name, M) in enumerate(steps.items()):
    result = cv2.warpAffine(image, M, (w, h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(128, 128, 128))
    axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[i].set_title(name, fontsize=10)
    axes[i].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '04_combine_affine.png'), dpi=150, bbox_inches='tight')
plt.show()

print("组合变换矩阵:")
print("M = T2 @ R @ S @ T1")
print(f"\n[{M_combined[0, 0]:.4f}  {M_combined[0, 1]:.4f}  {M_combined[0, 2]:.2f}]")
print(f"[{M_combined[1, 0]:.4f}  {M_combined[1, 1]:.4f}  {M_combined[1, 2]:.2f}]")
print(f"[{M_combined[2, 0]:.4f}  {M_combined[2, 1]:.4f}  {M_combined[2, 2]:.2f}]")
