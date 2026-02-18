"""
示例2：对比度限制的原理
- 演示clip limit如何截断直方图
- 截断部分均匀分配给其他灰度级
- 不同clip limit值的效果对比
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_clip_limit():
    """演示对比度限制的作用"""
    hist = np.array([100, 50, 200, 800, 150, 80, 60, 40], dtype=np.float32)

    clip_limits = [200, 400, 800]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('对比度限制（Clip Limit）原理', fontsize=14, fontweight='bold')

    # 原始直方图
    axes[0].bar(range(len(hist)), hist, color='blue', alpha=0.7)
    axes[0].set_title('原始直方图', fontsize=11)
    axes[0].set_ylim([0, 900])
    axes[0].axhline(y=800, color='r', linestyle='--', label='最大值')
    axes[0].legend()

    for i, clip in enumerate(clip_limits, 1):
        clipped_hist = hist.copy()

        # 计算超出部分
        excess = 0
        for j, val in enumerate(clipped_hist):
            if val > clip:
                excess += val - clip
                clipped_hist[j] = clip

        # 均匀分配超出部分
        redistribute = excess / len(clipped_hist)
        clipped_hist += redistribute

        axes[i].bar(range(len(clipped_hist)), clipped_hist, color='green', alpha=0.7)
        axes[i].axhline(y=clip, color='r', linestyle='--', label=f'限制={clip}')
        axes[i].set_title(f'Clip Limit = {clip}', fontsize=11)
        axes[i].set_ylim([0, 900])
        axes[i].legend()

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '02_clip_limit_principle.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("对比度限制（Clip Limit）的作用：")
    print("- 较小的clip limit → 对比度增强较弱，但更稳定")
    print("- 较大的clip limit → 对比度增强较强，接近普通均衡化")
    print("- 默认值通常为2.0-4.0")


demonstrate_clip_limit()
