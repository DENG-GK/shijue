"""
示例3：可视化伽马曲线
- 绘制不同γ值的变换曲线
- γ<1在曲线上方（提亮），γ>1在曲线下方（压暗）
- 渐变条直观展示各γ值的视觉效果
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_gamma_curves():
    """可视化不同γ值的伽马变换曲线"""
    x = np.linspace(0, 1, 256)
    gamma_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('伽马变换曲线可视化', fontsize=14, fontweight='bold')

    # 变换曲线
    colors = plt.cm.viridis(np.linspace(0, 1, len(gamma_values)))

    for gamma, color in zip(gamma_values, colors):
        y = np.power(x, gamma)
        axes[0].plot(x, y, color=color, label=f'γ={gamma}', linewidth=2)

    axes[0].set_xlabel('输入像素值', fontsize=12)
    axes[0].set_ylabel('输出像素值', fontsize=12)
    axes[0].set_title('伽马变换曲线', fontsize=13)
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # 渐变条效果展示
    input_gradient = np.tile(np.linspace(0, 255, 256), (50, 1)).astype(np.uint8)
    gamma_demo = [0.3, 0.6, 1.0, 1.5, 3.0]
    gradient_stack = [input_gradient]

    for gamma in gamma_demo:
        table = np.array([((i / 255.0) ** gamma) * 255
                          for i in range(256)]).astype(np.uint8)
        transformed = table[input_gradient]
        gradient_stack.append(transformed)

    combined = np.vstack(gradient_stack)
    axes[1].imshow(combined, cmap='gray', aspect='auto')
    axes[1].set_yticks([25, 75, 125, 175, 225, 275])
    axes[1].set_yticklabels(['输入', 'γ=0.3', 'γ=0.6', 'γ=1.0', 'γ=1.5', 'γ=3.0'])
    axes[1].set_xlabel('灰度级', fontsize=12)
    axes[1].set_title('各γ值的渐变效果', fontsize=13)

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '03_gamma_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()


visualize_gamma_curves()
print("伽马曲线特点：")
print("- γ<1: 曲线在对角线上方，整体提亮")
print("- γ=1: 对角线，无变化")
print("- γ>1: 曲线在对角线下方，整体压暗")
