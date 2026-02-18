"""
示例7：逆变换
- 正变换→逆变换恢复原图
- np.linalg.inv() 矩阵求逆
- M @ M_inv ≈ I 验证
- 恢复误差分析
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试图像
image = np.zeros((200, 300, 3), dtype=np.uint8)
image[:, :] = [200, 200, 200]
cv2.rectangle(image, (50, 50), (250, 150), (0, 128, 255), -1)
cv2.putText(image, 'INVERSE', (70, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

h, w = image.shape[:2]
center = (w // 2, h // 2)

# 构建正变换（旋转+缩放+平移）
angle = 30
scale = 0.8
tx, ty = 50, 30

M_rotate = cv2.getRotationMatrix2D(center, angle, scale)
M_rotate[0, 2] += tx
M_rotate[1, 2] += ty

# 转为3x3
M_3x3 = np.vstack([M_rotate, [0, 0, 1]])

# 正变换
transformed = cv2.warpAffine(image, M_rotate, (w, h))

# 求逆矩阵
M_inv_3x3 = np.linalg.inv(M_3x3)
M_inv = M_inv_3x3[:2, :]

# 逆变换恢复
recovered = cv2.warpAffine(transformed, M_inv, (w, h))

# 误差
error = cv2.absdiff(image, recovered)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('逆变换演示', fontsize=14, fontweight='bold')

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('正变换')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(recovered, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('逆变换恢复')
axes[0, 2].axis('off')

axes[0, 3].imshow(cv2.cvtColor(error * 10, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(f'误差 (10x)\n最大: {error.max()}')
axes[0, 3].axis('off')

# 正变换矩阵
text = "正变换矩阵 M:\n"
text += f"[{M_rotate[0, 0]:.3f}  {M_rotate[0, 1]:.3f}  {M_rotate[0, 2]:.1f}]\n"
text += f"[{M_rotate[1, 0]:.3f}  {M_rotate[1, 1]:.3f}  {M_rotate[1, 2]:.1f}]"
axes[1, 0].text(0.1, 0.5, text, fontsize=11, family='monospace',
                verticalalignment='center', transform=axes[1, 0].transAxes)
axes[1, 0].axis('off')
axes[1, 0].set_title('正变换矩阵')

# 逆变换矩阵
text = "逆变换矩阵 M⁻¹:\n"
text += f"[{M_inv[0, 0]:.3f}  {M_inv[0, 1]:.3f}  {M_inv[0, 2]:.1f}]\n"
text += f"[{M_inv[1, 0]:.3f}  {M_inv[1, 1]:.3f}  {M_inv[1, 2]:.1f}]"
axes[1, 1].text(0.1, 0.5, text, fontsize=11, family='monospace',
                verticalalignment='center', transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')
axes[1, 1].set_title('逆变换矩阵')

# 验证 M @ M_inv = I
identity = M_3x3 @ M_inv_3x3
text = "M @ M⁻¹ (应为单位矩阵):\n"
for i in range(3):
    text += f"[{identity[i, 0]:.4f}  {identity[i, 1]:.4f}  {identity[i, 2]:.4f}]\n"
axes[1, 2].text(0.1, 0.5, text, fontsize=10, family='monospace',
                verticalalignment='center', transform=axes[1, 2].transAxes)
axes[1, 2].axis('off')
axes[1, 2].set_title('验证 M×M⁻¹=I')

axes[1, 3].axis('off')

plt.tight_layout()
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, '07_inverse_transform.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"逆变换恢复结果:")
print(f"  最大误差: {error.max()}")
print(f"  平均误差: {error.mean():.4f}")
