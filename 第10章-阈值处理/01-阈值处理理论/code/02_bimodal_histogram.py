"""
示例2：双峰直方图分析
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_histogram(image, title="Image"):
    """分析图像直方图，找出可能的阈值"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=5)

    peaks = []
    for i in range(1, 255):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            if hist_smooth[i] > np.mean(hist_smooth):
                peaks.append(i)

    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.imshow(gray, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.subplot(122)
    plt.plot(hist, alpha=0.5, label='Original')
    plt.plot(hist_smooth, label='Smoothed')
    for p in peaks:
        plt.axvline(x=p, color='r', linestyle='--', alpha=0.5)
        plt.annotate(f'Peak: {p}', (p, hist_smooth[p]))
    plt.title('Histogram Analysis')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

    if len(peaks) >= 2:
        suggested_threshold = (peaks[0] + peaks[1]) // 2
        print(f"检测到双峰: {peaks[:2]}")
        print(f"建议阈值: {suggested_threshold}")
        return suggested_threshold
    else:
        print("未检测到明显的双峰分布")
        return 128

test_img = np.zeros((300, 300), dtype=np.uint8)
test_img[:, :150] = np.random.normal(60, 15, (300, 150)).clip(0, 255)
test_img[:, 150:] = np.random.normal(180, 15, (300, 150)).clip(0, 255)
test_img = test_img.astype(np.uint8)

threshold = analyze_histogram(test_img, "Bimodal Test Image")
