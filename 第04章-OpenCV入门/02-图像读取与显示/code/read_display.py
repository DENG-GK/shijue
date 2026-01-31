# -*- coding: utf-8 -*-
"""
图像读取与显示 - 示例代码
先运行 generate_images.py 生成测试图像
"""
import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "..", "images")

print("=" * 50)
print("OpenCV 图像读取与显示")
print("=" * 50)

# ========== 1. 读取彩色图像 ==========
print("\n【1. 读取彩色图像】")
img_path = os.path.join(IMAGES_DIR, "test_color.png")
if os.path.exists(img_path):
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(f"彩色图: shape={img_color.shape}, dtype={img_color.dtype}")
else:
    # 没有测试图时创建一个
    img_color = np.zeros((200, 300, 3), dtype=np.uint8)
    img_color[:, :100] = [255, 0, 0]    # 蓝
    img_color[:, 100:200] = [0, 255, 0]  # 绿
    img_color[:, 200:] = [0, 0, 255]     # 红
    print(f"生成测试图: shape={img_color.shape}")

# ========== 2. 读取灰度图 ==========
print("\n【2. 读取灰度图】")
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(img_path) else \
    cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print(f"灰度图: shape={img_gray.shape}, dtype={img_gray.dtype}")

# ========== 3. 显示图像 ==========
print("\n【3. 显示图像】")
print("使用 cv2.imshow() 显示（按任意键关闭）:")
cv2.imshow("Color Image", img_color)
cv2.imshow("Gray Image", img_gray)
cv2.waitKey(2000)  # 等待2秒或按键
cv2.destroyAllWindows()
print("窗口已关闭")

# ========== 4. matplotlib 显示 ==========
print("\n【4. matplotlib 显示】")
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # BGR → RGB 转换
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Color (RGB)")
    axes[0].axis("off")

    axes[1].imshow(img_gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    output_path = os.path.join(IMAGES_DIR, "display_result.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"matplotlib结果已保存: {output_path}")
except ImportError:
    print("matplotlib未安装，跳过")

print("\n示例代码运行完成！")
