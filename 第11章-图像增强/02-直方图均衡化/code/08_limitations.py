"""
示例8：直方图均衡化的局限性
- 噪声放大问题：椒盐噪声在均衡化后被显著增强
- 过度增强：对正常图像进行均衡化可能过度处理
- 局部对比度问题：全局方法忽略局部亮度差异
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_limitations():
    """展示直方图均衡化的局限性"""

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('直方图均衡化的局限性', fontsize=14, fontweight='bold')

    # 1. 噪声放大
    noisy = np.random.normal(128, 10, (200, 300)).clip(0, 255).astype(np.uint8)
    salt_pepper = noisy.copy()
    noise_mask = np.random.random(noisy.shape)
    salt_pepper[noise_mask < 0.02] = 0
    salt_pepper[noise_mask > 0.98] = 255

    noisy_eq = cv2.equalizeHist(salt_pepper)

    axes[0, 0].imshow(salt_pepper, cmap='gray')
    axes[0, 0].set_title('含噪声的图像', fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_eq, cmap='gray')
    axes[0, 1].set_title('均衡化后（噪声被放大!）', fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].text(0.5, 0.5, '问题1:\n噪声被放大\n椒盐噪声更明显',
                    ha='center', va='center', fontsize=13, color='red',
                    transform=axes[0, 2].transAxes)
    axes[0, 2].axis('off')

    # 2. 过度增强
    normal = np.random.normal(128, 40, (200, 300)).clip(0, 255).astype(np.uint8)
    cv2.rectangle(normal, (50, 50), (250, 150), 80, -1)
    normal_eq = cv2.equalizeHist(normal)

    axes[1, 0].imshow(normal, cmap='gray')
    axes[1, 0].set_title('正常图像（对比度足够）', fontsize=10)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(normal_eq, cmap='gray')
    axes[1, 1].set_title('均衡化后（过度增强）', fontsize=10)
    axes[1, 1].axis('off')

    axes[1, 2].text(0.5, 0.5, '问题2:\n过度增强\n对正常图像不适用',
                    ha='center', va='center', fontsize=13, color='red',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')

    # 3. 局部对比度问题
    local_issue = np.zeros((200, 300), dtype=np.uint8)
    local_issue[:, :150] = 50
    local_issue[:, 150:] = 200
    cv2.circle(local_issue, (75, 100), 30, 80, -1)
    cv2.circle(local_issue, (225, 100), 30, 170, -1)

    local_eq = cv2.equalizeHist(local_issue)

    axes[2, 0].imshow(local_issue, cmap='gray')
    axes[2, 0].set_title('局部亮度不均匀', fontsize=10)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(local_eq, cmap='gray')
    axes[2, 1].set_title('全局均衡化结果', fontsize=10)
    axes[2, 1].axis('off')

    axes[2, 2].text(0.5, 0.5, '问题3:\n全局方法忽略\n局部对比度差异\n→ 使用CLAHE解决',
                    ha='center', va='center', fontsize=12, color='red',
                    transform=axes[2, 2].transAxes)
    axes[2, 2].axis('off')

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '08_limitations.png'), dpi=150, bbox_inches='tight')
    plt.show()


demonstrate_limitations()

print("\n直方图均衡化的缺点：")
print("=" * 50)
print("1. 噪声放大：噪声会被当作有效信息增强")
print("2. 过度增强：对已经正常的图像可能过度处理")
print("3. 全局方法：忽略局部对比度差异")
print("4. 不可控：无法调节增强程度")
print("=" * 50)
print("\n解决方案：使用CLAHE（自适应直方图均衡化）")
