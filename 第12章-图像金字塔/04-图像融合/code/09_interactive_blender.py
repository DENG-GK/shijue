"""
示例9：交互式融合工具
- OpenCV trackbar调节levels/position/feather
- 实时预览融合效果
- 快捷键：q退出 s保存
"""
import cv2
import numpy as np
import os


def build_gaussian_pyramid(image, levels):
    pyramid = [image.astype(np.float64)]
    current = image.astype(np.float64)
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(image, levels):
    G = build_gaussian_pyramid(image, levels)
    L = []
    for i in range(levels - 1):
        expanded = cv2.pyrUp(G[i + 1])
        if expanded.shape != G[i].shape:
            expanded = cv2.resize(expanded, (G[i].shape[1], G[i].shape[0]))
        L.append(G[i] - expanded)
    L.append(G[-1])
    return L


def reconstruct_from_laplacian(pyramid):
    result = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        expanded = cv2.pyrUp(result)
        if expanded.shape != pyramid[i].shape:
            expanded = cv2.resize(expanded, (pyramid[i].shape[1], pyramid[i].shape[0]))
        result = expanded + pyramid[i]
    return result


def pyramid_blend(img1, img2, mask, levels=6):
    L1 = build_laplacian_pyramid(img1, levels)
    L2 = build_laplacian_pyramid(img2, levels)
    GM = build_gaussian_pyramid(mask, levels)
    L_blend = []
    for l1, l2, gm in zip(L1, L2, GM):
        if len(l1.shape) == 3 and len(gm.shape) == 2:
            gm = np.stack([gm] * 3, axis=2)
        L_blend.append(l1 * gm + l2 * (1 - gm))
    return np.clip(reconstruct_from_laplacian(L_blend), 0, 255).astype(np.uint8)


def create_interactive_blender():
    """交互式融合工具"""
    size = (300, 400)

    img1 = np.zeros((*size, 3), dtype=np.uint8)
    img1[:, :] = [0, 200, 255]
    cv2.rectangle(img1, (50, 50), (200, 200), (0, 150, 200), -1)
    cv2.putText(img1, 'IMAGE 1', (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    img2 = np.zeros((*size, 3), dtype=np.uint8)
    img2[:, :] = [255, 100, 0]
    cv2.circle(img2, (300, 150), 80, (200, 80, 0), -1)
    cv2.putText(img2, 'IMAGE 2', (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    params = {'levels': 5, 'pos': size[1] // 2, 'feather': 30}

    def update(x=None):
        mask = np.zeros(size, dtype=np.float64)
        pos = params['pos']
        f = max(1, params['feather'])
        for i in range(size[1]):
            if i < pos - f:
                mask[:, i] = 1.0
            elif i > pos + f:
                mask[:, i] = 0.0
            else:
                mask[:, i] = 0.5 * (1 + np.cos(np.pi * (i - pos + f) / (2 * f)))

        blended = pyramid_blend(img1, img2, mask, max(1, params['levels']))
        mask_vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        display = np.vstack([np.hstack([img1, img2]),
                             np.hstack([blended, mask_vis])])

        cv2.putText(display, f"Levels: {params['levels']}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow('Pyramid Blender', display)

    cv2.namedWindow('Pyramid Blender', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Levels', 'Pyramid Blender', 5, 8,
                       lambda v: (params.update({'levels': max(1, v)}), update()))
    cv2.createTrackbar('Position', 'Pyramid Blender', size[1] // 2, size[1],
                       lambda v: (params.update({'pos': v}), update()))
    cv2.createTrackbar('Feather', 'Pyramid Blender', 30, 100,
                       lambda v: (params.update({'feather': max(1, v)}), update()))
    update()

    print("交互式融合工具")
    print("  Levels: 金字塔层数")
    print("  Position: 融合边界位置")
    print("  Feather: 过渡宽度")
    print("  q=退出 s=保存")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
            os.makedirs(save_dir, exist_ok=True)
            print("已保存")
    cv2.destroyAllWindows()


# 取消注释运行交互工具
# create_interactive_blender()

print("交互式融合工具代码已就绪")
print("取消最后一行注释即可运行")
