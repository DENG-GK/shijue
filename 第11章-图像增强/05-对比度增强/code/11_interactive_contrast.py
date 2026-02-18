"""
示例11：交互式对比度调整工具
- OpenCV trackbar实时调节
- 支持Alpha(对比度)、Beta(亮度)、Gamma、CLAHE
- 信息面板显示参数和统计量
- 快捷键：r重置 s保存 q退出
"""
import cv2
import numpy as np
import os


def create_contrast_adjustment_tool():
    """交互式对比度调整工具"""
    # 创建测试图像
    image = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (250, 250), (80, 100, 120), -1)
    cv2.circle(image, (430, 200), 80, (150, 170, 190), -1)
    cv2.putText(image, 'TEST', (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3)

    cv2.namedWindow('Contrast Adjustment', cv2.WINDOW_NORMAL)

    params = {
        'alpha': 100,      # 对比度 (100 = 1.0)
        'beta': 128,       # 亮度 (128 = 0偏移)
        'gamma': 100,      # Gamma (100 = 1.0)
        'clip_limit': 20,  # CLAHE clipLimit (20 = 2.0)
    }

    # 保存增强结果用于save
    current_result = [image.copy()]

    def update_image(x=None):
        alpha = params['alpha'] / 100.0
        beta = params['beta'] - 128
        gamma_val = max(params['gamma'], 1) / 100.0
        clip_limit = params['clip_limit'] / 10.0

        # 对比度和亮度调整
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Gamma校正
        if abs(gamma_val - 1.0) > 0.01:
            inv_gamma = 1.0 / gamma_val
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                              for i in range(256)]).astype(np.uint8)
            adjusted = cv2.LUT(adjusted, table)

        # CLAHE
        if clip_limit > 0.1:
            lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            adjusted = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        current_result[0] = adjusted.copy()

        # 信息面板
        info_height = 80
        info_panel = np.zeros((info_height, adjusted.shape[1], 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(info_panel, f'Alpha: {alpha:.2f}', (10, 25), font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f'Beta: {beta}', (180, 25), font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f'Gamma: {gamma_val:.2f}', (310, 25), font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f'CLAHE: {clip_limit:.1f}', (460, 25), font, 0.5, (255, 255, 255), 1)

        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        cv2.putText(info_panel, f'Mean: {np.mean(gray):.1f}', (10, 55), font, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, f'Std: {np.std(gray):.1f}', (180, 55), font, 0.5, (200, 200, 200), 1)
        cv2.putText(info_panel, f'[R]eset [S]ave [Q]uit', (310, 55), font, 0.5, (150, 200, 255), 1)

        display = np.vstack([adjusted, info_panel])
        cv2.imshow('Contrast Adjustment', display)

    def on_alpha(val):
        params['alpha'] = max(1, val)
        update_image()

    def on_beta(val):
        params['beta'] = val
        update_image()

    def on_gamma(val):
        params['gamma'] = max(1, val)
        update_image()

    def on_clahe(val):
        params['clip_limit'] = val
        update_image()

    cv2.createTrackbar('Alpha', 'Contrast Adjustment', 100, 300, on_alpha)
    cv2.createTrackbar('Beta', 'Contrast Adjustment', 128, 255, on_beta)
    cv2.createTrackbar('Gamma', 'Contrast Adjustment', 100, 300, on_gamma)
    cv2.createTrackbar('CLAHE', 'Contrast Adjustment', 20, 100, on_clahe)

    update_image()

    print("交互式对比度调整工具")
    print("=" * 40)
    print("  Alpha：对比度（100 = 不变）")
    print("  Beta：亮度（128 = 不变）")
    print("  Gamma：伽马校正（100 = 不变）")
    print("  CLAHE：自适应均衡化 clipLimit")
    print("快捷键：r=重置 s=保存 q=退出")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.setTrackbarPos('Alpha', 'Contrast Adjustment', 100)
            cv2.setTrackbarPos('Beta', 'Contrast Adjustment', 128)
            cv2.setTrackbarPos('Gamma', 'Contrast Adjustment', 100)
            cv2.setTrackbarPos('CLAHE', 'Contrast Adjustment', 20)
        elif key == ord('s'):
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, '11_interactive_result.png')
            cv2.imwrite(save_path, current_result[0])
            print(f"已保存到: {save_path}")

    cv2.destroyAllWindows()


# 运行交互工具（取消注释使用）
# create_contrast_adjustment_tool()

print("交互式对比度调整工具代码已就绪")
print("取消最后一行注释即可运行交互式工具")
