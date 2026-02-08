# HOG特征

## 1. HOG特征概述

哼，笨蛋，HOG（Histogram of Oriented Gradients，方向梯度直方图）可是目标检测领域的经典算法！本小姐今天就来给你好好讲讲！(￣▽￣)／

### 1.1 HOG的诞生

HOG由Navneet Dalal和Bill Triggs在2005年提出，最初用于行人检测。它通过统计图像局部区域的梯度方向分布来描述物体外观和形状。

### 1.2 HOG vs 点特征

| 特性 | HOG | SIFT/ORB |
|------|-----|----------|
| 类型 | 密集描述符 | 稀疏描述符 |
| 应用 | 物体检测 | 特征匹配 |
| 计算单位 | 固定窗口 | 关键点 |
| 尺度不变性 | 需要多尺度扫描 | 内置 |
| 典型用途 | 行人检测、车辆检测 | 图像拼接、物体识别 |

### 1.3 HOG的核心思想

图像中物体的局部外观和形状可以通过局部梯度的分布来描述：
1. 将图像划分为小的连通区域（细胞单元cell）
2. 计算每个cell内的梯度方向直方图
3. 将多个cell组合成块（block）进行归一化
4. 拼接所有block的特征形成最终描述符

## 2. HOG特征计算流程

### 2.1 计算步骤

```
原始图像
    ↓
灰度化 + 预处理
    ↓
计算梯度（幅值和方向）
    ↓
划分Cell（如8×8像素）
    ↓
构建梯度方向直方图
    ↓
Block归一化（如2×2 cells）
    ↓
拼接所有Block特征
    ↓
HOG特征向量
```

### 2.2 代码示例1：理解梯度计算

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
cv2.circle(img, (100, 100), 30, 100, -1)

# 计算梯度
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

# 计算幅值和方向
magnitude = np.sqrt(gx**2 + gy**2)
orientation = np.arctan2(gy, gx) * 180 / np.pi  # 转换为度数

# 将方向转换到0-180度（无符号梯度）
orientation = np.where(orientation < 0, orientation + 180, orientation)

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(gx, cmap='coolwarm')
axes[0, 1].set_title('X方向梯度 (Gx)')
axes[0, 1].axis('off')

axes[0, 2].imshow(gy, cmap='coolwarm')
axes[0, 2].set_title('Y方向梯度 (Gy)')
axes[0, 2].axis('off')

im = axes[1, 0].imshow(magnitude, cmap='hot')
axes[1, 0].set_title('梯度幅值')
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

im = axes[1, 1].imshow(orientation, cmap='hsv')
axes[1, 1].set_title('梯度方向 (0-180°)')
axes[1, 1].axis('off')
plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

# 绘制梯度场
step = 10
Y, X = np.mgrid[step//2:200:step, step//2:200:step]
U = gx[step//2::step, step//2::step]
V = gy[step//2::step, step//2::step]

axes[1, 2].imshow(img, cmap='gray', alpha=0.5)
axes[1, 2].quiver(X, Y, U, -V, color='red', scale=1000)
axes[1, 2].set_title('梯度场可视化')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('hog_gradient.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"梯度幅值范围: {magnitude.min():.2f} - {magnitude.max():.2f}")
print(f"梯度方向范围: {orientation.min():.2f}° - {orientation.max():.2f}°")
```

### 2.3 代码示例2：Cell直方图构建

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_cell_histogram(magnitude, orientation, num_bins=9):
    """计算单个cell的梯度方向直方图"""
    bin_width = 180 / num_bins  # 每个bin的角度范围
    histogram = np.zeros(num_bins)

    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            angle = orientation[i, j]
            mag = magnitude[i, j]

            # 确定所属的bin
            bin_idx = int(angle / bin_width) % num_bins

            # 双线性插值（简化版：直接加入）
            histogram[bin_idx] += mag

    return histogram

# 创建测试图像
img = np.zeros((64, 64), dtype=np.uint8)
cv2.line(img, (10, 32), (54, 32), 255, 3)  # 水平线
cv2.line(img, (32, 10), (32, 54), 255, 3)  # 垂直线

# 计算梯度
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
magnitude = np.sqrt(gx**2 + gy**2)
orientation = np.arctan2(gy, gx) * 180 / np.pi
orientation = np.where(orientation < 0, orientation + 180, orientation)

# 将图像划分为8x8的cells
cell_size = 8
num_cells_x = img.shape[1] // cell_size
num_cells_y = img.shape[0] // cell_size

# 计算每个cell的直方图
histograms = []
for cy in range(num_cells_y):
    row = []
    for cx in range(num_cells_x):
        cell_mag = magnitude[cy*cell_size:(cy+1)*cell_size,
                            cx*cell_size:(cx+1)*cell_size]
        cell_ori = orientation[cy*cell_size:(cy+1)*cell_size,
                              cx*cell_size:(cx+1)*cell_size]
        hist = compute_cell_histogram(cell_mag, cell_ori)
        row.append(hist)
    histograms.append(row)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像 (64x64)')
# 绘制cell网格
for i in range(1, num_cells_x):
    axes[0, 0].axvline(x=i*cell_size, color='red', linewidth=0.5)
for i in range(1, num_cells_y):
    axes[0, 0].axhline(y=i*cell_size, color='red', linewidth=0.5)
axes[0, 0].axis('off')

axes[0, 1].imshow(magnitude, cmap='hot')
axes[0, 1].set_title('梯度幅值')
axes[0, 1].axis('off')

# 显示中心cell的直方图
center_hist = histograms[4][4]
bins = np.arange(9) * 20 + 10  # 0-180度，每20度一个bin
axes[1, 0].bar(bins, center_hist, width=15, edgecolor='black')
axes[1, 0].set_xlabel('梯度方向 (度)')
axes[1, 0].set_ylabel('梯度幅值累加')
axes[1, 0].set_title('中心Cell的方向直方图')
axes[1, 0].set_xticks(bins)

# 可视化所有cells的主方向
dominant_orientations = np.zeros((num_cells_y, num_cells_x))
for cy in range(num_cells_y):
    for cx in range(num_cells_x):
        hist = histograms[cy][cx]
        if np.sum(hist) > 0:
            dominant_orientations[cy, cx] = np.argmax(hist) * 20 + 10

im = axes[1, 1].imshow(dominant_orientations, cmap='hsv', vmin=0, vmax=180)
axes[1, 1].set_title('各Cell的主导方向')
plt.colorbar(im, ax=axes[1, 1], label='角度(度)')

plt.tight_layout()
plt.savefig('hog_cell_histogram.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 3. OpenCV中的HOG

### 3.1 cv2.HOGDescriptor

OpenCV提供了`HOGDescriptor`类来计算HOG特征：

```python
hog = cv2.HOGDescriptor(
    winSize,        # 检测窗口大小
    blockSize,      # Block大小
    blockStride,    # Block滑动步长
    cellSize,       # Cell大小
    nbins           # 方向bin数量
)
```

### 3.2 代码示例3：使用HOGDescriptor

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像
img = np.zeros((128, 64), dtype=np.uint8)
# 模拟简单人形
cv2.ellipse(img, (32, 20), (12, 15), 0, 0, 360, 200, -1)  # 头
cv2.rectangle(img, (22, 35), (42, 80), 180, -1)  # 身体
cv2.rectangle(img, (20, 80), (30, 120), 160, -1)  # 左腿
cv2.rectangle(img, (34, 80), (44, 120), 160, -1)  # 右腿
cv2.rectangle(img, (10, 40), (22, 70), 140, -1)  # 左臂
cv2.rectangle(img, (42, 40), (54, 70), 140, -1)  # 右臂

# 创建HOG描述符
# 标准行人检测配置：64x128窗口
hog = cv2.HOGDescriptor(
    (64, 128),   # winSize
    (16, 16),    # blockSize
    (8, 8),      # blockStride
    (8, 8),      # cellSize
    9            # nbins
)

# 计算HOG特征
descriptor = hog.compute(img)

print(f"图像尺寸: {img.shape}")
print(f"HOG特征维度: {descriptor.shape}")
print(f"特征长度: {len(descriptor)}")

# 计算特征维度的公式：
# blocks_x = (winSize[0] - blockSize[0]) / blockStride[0] + 1
# blocks_y = (winSize[1] - blockSize[1]) / blockStride[1] + 1
# cells_per_block = (blockSize[0] / cellSize[0]) * (blockSize[1] / cellSize[1])
# 特征长度 = blocks_x * blocks_y * cells_per_block * nbins

blocks_x = (64 - 16) // 8 + 1  # = 7
blocks_y = (128 - 16) // 8 + 1  # = 15
cells_per_block = (16 // 8) * (16 // 8)  # = 4
expected_length = blocks_x * blocks_y * cells_per_block * 9

print(f"\n理论计算:")
print(f"  Blocks (X): {blocks_x}")
print(f"  Blocks (Y): {blocks_y}")
print(f"  Cells per Block: {cells_per_block}")
print(f"  预期特征长度: {expected_length}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('输入图像 (64x128)')
axes[0].axis('off')

# 特征直方图
axes[1].plot(descriptor[:100], 'b-', linewidth=0.5)
axes[1].set_xlabel('特征索引')
axes[1].set_ylabel('特征值')
axes[1].set_title('HOG特征（前100维）')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hog_descriptor.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.3 代码示例4：HOG特征可视化

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# 创建测试图像
img = np.zeros((128, 64), dtype=np.uint8)
# 绘制简单人形
cv2.ellipse(img, (32, 20), (12, 15), 0, 0, 360, 200, -1)
cv2.rectangle(img, (22, 35), (42, 80), 180, -1)
cv2.rectangle(img, (20, 80), (30, 120), 160, -1)
cv2.rectangle(img, (34, 80), (44, 120), 160, -1)
cv2.rectangle(img, (10, 40), (22, 70), 140, -1)
cv2.rectangle(img, (42, 40), (54, 70), 140, -1)

# 使用skimage计算HOG（便于可视化）
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True,
                    block_norm='L2-Hys')

# 增强HOG可视化
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(hog_image, cmap='gray')
axes[1].set_title('HOG特征可视化')
axes[1].axis('off')

axes[2].imshow(hog_image_rescaled, cmap='gray')
axes[2].set_title('HOG特征可视化（增强）')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('hog_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"HOG特征维度: {fd.shape}")
```

## 4. HOG行人检测

### 4.1 代码示例5：使用预训练的行人检测器

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建包含"行人"的测试场景
def create_pedestrian_scene():
    scene = np.zeros((400, 600), dtype=np.uint8)
    scene[:] = 100  # 背景

    # 添加简化的人形（多个位置）
    def draw_person(img, x, y, scale=1.0):
        h = int(100 * scale)
        w = int(40 * scale)
        # 头
        cv2.circle(img, (x, y - int(h*0.4)), int(w*0.3), 200, -1)
        # 身体
        cv2.rectangle(img, (x - int(w*0.3), y - int(h*0.3)),
                     (x + int(w*0.3), y + int(h*0.2)), 180, -1)
        # 腿
        cv2.rectangle(img, (x - int(w*0.25), y + int(h*0.2)),
                     (x - int(w*0.05), y + int(h*0.5)), 160, -1)
        cv2.rectangle(img, (x + int(w*0.05), y + int(h*0.2)),
                     (x + int(w*0.25), y + int(h*0.5)), 160, -1)

    draw_person(scene, 150, 250, 1.0)
    draw_person(scene, 350, 280, 0.8)
    draw_person(scene, 500, 240, 1.1)

    return scene

scene = create_pedestrian_scene()

# 创建HOG检测器并设置预训练的行人检测器
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 检测行人
# detectMultiScale返回检测框和权重
boxes, weights = hog.detectMultiScale(
    scene,
    winStride=(8, 8),      # 窗口滑动步长
    padding=(4, 4),        # 填充
    scale=1.05,            # 尺度缩放因子
    useMeanshiftGrouping=False
)

print(f"检测到 {len(boxes)} 个目标")

# 绘制检测结果
scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

for i, (x, y, w, h) in enumerate(boxes):
    cv2.rectangle(scene_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(scene_color, f'{weights[i]:.2f}', (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(scene, cmap='gray')
plt.title('原始场景')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(scene_color, cv2.COLOR_BGR2RGB))
plt.title(f'HOG行人检测结果 (检测到 {len(boxes)} 个)')
plt.axis('off')

plt.tight_layout()
plt.savefig('hog_pedestrian_detection.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.2 代码示例6：自定义HOG参数检测

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像
img = np.zeros((300, 400), dtype=np.uint8)
img[:] = 80

# 添加一些矩形目标
cv2.rectangle(img, (50, 50), (120, 180), 200, -1)
cv2.rectangle(img, (180, 80), (260, 220), 180, -1)
cv2.rectangle(img, (300, 40), (380, 200), 220, -1)

# 比较不同参数设置
configs = [
    {'winStride': (4, 4), 'scale': 1.02},
    {'winStride': (8, 8), 'scale': 1.05},
    {'winStride': (16, 16), 'scale': 1.1},
    {'winStride': (8, 8), 'scale': 1.2},
]

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, cfg in enumerate(configs):
    ax = axes[idx // 2, idx % 2]

    # 检测
    boxes, weights = hog.detectMultiScale(
        img,
        winStride=cfg['winStride'],
        scale=cfg['scale']
    )

    # 绘制
    img_result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    ax.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    ax.set_title(f"winStride={cfg['winStride']}, scale={cfg['scale']}\n"
                 f"检测数: {len(boxes)}")
    ax.axis('off')

plt.suptitle('HOG检测参数对比', fontsize=14)
plt.tight_layout()
plt.savefig('hog_params_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.3 代码示例7：非极大值抑制(NMS)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def non_max_suppression(boxes, scores, threshold=0.5):
    """非极大值抑制"""
    if len(boxes) == 0:
        return []

    # 转换为浮点数
    boxes = boxes.astype(float)

    # 获取坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # 计算面积
    areas = (x2 - x1) * (y2 - y1)

    # 按分数排序
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 计算IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # 保留IoU小于阈值的框
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep

# 创建测试场景
scene = np.zeros((400, 500), dtype=np.uint8)
scene[:] = 100

# 添加目标
cv2.rectangle(scene, (100, 100), (200, 300), 200, -1)
cv2.rectangle(scene, (300, 80), (420, 320), 180, -1)

# HOG检测（可能产生重叠框）
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

boxes, weights = hog.detectMultiScale(
    scene,
    winStride=(4, 4),
    scale=1.02,
    hitThreshold=0
)

print(f"NMS前检测框数量: {len(boxes)}")

# 应用NMS
if len(boxes) > 0:
    keep_indices = non_max_suppression(boxes, weights, threshold=0.3)
    boxes_nms = boxes[keep_indices]
    weights_nms = weights[keep_indices]
    print(f"NMS后检测框数量: {len(boxes_nms)}")
else:
    boxes_nms = boxes
    weights_nms = weights

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(scene, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

# NMS前
scene_before = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in boxes:
    cv2.rectangle(scene_before, (x, y), (x+w, y+h), (0, 255, 0), 1)
axes[1].imshow(cv2.cvtColor(scene_before, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'NMS前 ({len(boxes)} 个框)')
axes[1].axis('off')

# NMS后
scene_after = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in boxes_nms:
    cv2.rectangle(scene_after, (x, y), (x+w, y+h), (0, 255, 0), 2)
axes[2].imshow(cv2.cvtColor(scene_after, cv2.COLOR_BGR2RGB))
axes[2].set_title(f'NMS后 ({len(boxes_nms)} 个框)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('hog_nms.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 5. 自定义HOG训练

### 5.1 代码示例8：提取HOG特征用于分类

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def create_dataset():
    """创建简单的形状数据集"""
    samples = []
    labels = []

    # 生成矩形样本
    for _ in range(50):
        img = np.zeros((64, 64), dtype=np.uint8)
        x = np.random.randint(10, 30)
        y = np.random.randint(10, 30)
        w = np.random.randint(20, 40)
        h = np.random.randint(20, 40)
        cv2.rectangle(img, (x, y), (x+w, y+h), 255, -1)
        samples.append(img)
        labels.append(0)

    # 生成圆形样本
    for _ in range(50):
        img = np.zeros((64, 64), dtype=np.uint8)
        x = np.random.randint(20, 44)
        y = np.random.randint(20, 44)
        r = np.random.randint(10, 20)
        cv2.circle(img, (x, y), r, 255, -1)
        samples.append(img)
        labels.append(1)

    return np.array(samples), np.array(labels)

# 创建数据集
X_images, y = create_dataset()

# 创建HOG描述符
hog = cv2.HOGDescriptor(
    (64, 64),    # winSize
    (16, 16),    # blockSize
    (8, 8),      # blockStride
    (8, 8),      # cellSize
    9            # nbins
)

# 提取HOG特征
X_hog = []
for img in X_images:
    descriptor = hog.compute(img)
    X_hog.append(descriptor.flatten())

X_hog = np.array(X_hog)
print(f"HOG特征形状: {X_hog.shape}")

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_hog, y, test_size=0.2, random_state=42
)

# 训练SVM分类器
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# 预测和评估
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n分类准确率: {accuracy:.2%}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['矩形', '圆形']))

# 可视化一些样本和预测结果
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    ax = axes[i // 5, i % 5]
    idx = np.random.randint(len(X_test))
    # 找到原始图像
    test_idx = np.where((X_hog == X_test[idx]).all(axis=1))[0][0]
    ax.imshow(X_images[test_idx], cmap='gray')
    true_label = '矩形' if y_test[idx] == 0 else '圆形'
    pred_label = '矩形' if y_pred[idx] == 0 else '圆形'
    color = 'green' if y_test[idx] == y_pred[idx] else 'red'
    ax.set_title(f'真:{true_label}\n预测:{pred_label}', color=color)
    ax.axis('off')

plt.suptitle(f'HOG+SVM形状分类 (准确率: {accuracy:.2%})', fontsize=14)
plt.tight_layout()
plt.savefig('hog_classification.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.2 代码示例9：训练自定义物体检测器

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_training_data():
    """创建训练数据（正样本和负样本）"""
    positive_samples = []
    negative_samples = []

    # 正样本：包含特定图案
    for _ in range(100):
        img = np.zeros((64, 128), dtype=np.uint8)
        # 绘制标准图案（模拟目标）
        cv2.rectangle(img, (20, 20), (44, 108), 200, -1)
        cv2.circle(img, (32, 40), 10, 255, -1)
        # 添加随机噪声
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        positive_samples.append(img)

    # 负样本：随机背景
    for _ in range(200):
        img = np.random.randint(50, 150, (64, 128), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        negative_samples.append(img)

    return positive_samples, negative_samples

# 创建训练数据
pos_samples, neg_samples = create_training_data()

# 创建HOG描述符
hog = cv2.HOGDescriptor(
    (64, 128),   # winSize
    (16, 16),    # blockSize
    (8, 8),      # blockStride
    (8, 8),      # cellSize
    9            # nbins
)

# 提取特征
def extract_features(samples, hog):
    features = []
    for sample in samples:
        descriptor = hog.compute(sample)
        features.append(descriptor.flatten())
    return np.array(features, dtype=np.float32)

pos_features = extract_features(pos_samples, hog)
neg_features = extract_features(neg_samples, hog)

print(f"正样本特征: {pos_features.shape}")
print(f"负样本特征: {neg_features.shape}")

# 准备训练数据
X = np.vstack([pos_features, neg_features])
y = np.hstack([np.ones(len(pos_features)), np.zeros(len(neg_features))])

# 训练SVM（使用OpenCV的SVM）
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(1.0)

# 训练
svm.train(X, cv2.ml.ROW_SAMPLE, y.astype(np.int32))

# 获取支持向量
sv = svm.getSupportVectors()
rho, _, _ = svm.getDecisionFunction(0)

print(f"\n支持向量数量: {len(sv)}")
print(f"决策边界 rho: {rho}")

# 测试检测器
def test_detector(test_img, hog, svm):
    descriptor = hog.compute(test_img)
    _, result = svm.predict(descriptor.reshape(1, -1).astype(np.float32))
    return result[0][0]

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 显示正样本
for i in range(4):
    axes[0, i].imshow(pos_samples[i], cmap='gray')
    pred = test_detector(pos_samples[i], hog, svm)
    axes[0, i].set_title(f'正样本\n预测: {"正" if pred == 1 else "负"}',
                        color='green' if pred == 1 else 'red')
    axes[0, i].axis('off')

# 显示负样本
for i in range(4):
    axes[1, i].imshow(neg_samples[i], cmap='gray')
    pred = test_detector(neg_samples[i], hog, svm)
    axes[1, i].set_title(f'负样本\n预测: {"正" if pred == 1 else "负"}',
                        color='red' if pred == 1 else 'green')
    axes[1, i].axis('off')

plt.suptitle('HOG+SVM自定义检测器训练结果', fontsize=14)
plt.tight_layout()
plt.savefig('hog_custom_detector.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 6. HOG性能优化

### 6.1 代码示例10：多尺度检测优化

```python
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def benchmark_hog_detection():
    """测试不同参数下的HOG检测性能"""

    # 创建测试图像
    img = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 添加一些形状
    cv2.rectangle(img, (100, 100), (200, 300), 220, -1)
    cv2.rectangle(img, (400, 150), (550, 400), 180, -1)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # 测试不同参数
    configs = [
        {'winStride': (4, 4), 'scale': 1.01, 'label': '高精度'},
        {'winStride': (8, 8), 'scale': 1.05, 'label': '平衡'},
        {'winStride': (16, 16), 'scale': 1.1, 'label': '高速'},
        {'winStride': (32, 32), 'scale': 1.2, 'label': '极速'},
    ]

    results = []

    for cfg in configs:
        times = []
        for _ in range(10):
            start = time.time()
            boxes, _ = hog.detectMultiScale(
                img,
                winStride=cfg['winStride'],
                scale=cfg['scale']
            )
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        fps = 1000 / avg_time

        results.append({
            'label': cfg['label'],
            'winStride': cfg['winStride'],
            'scale': cfg['scale'],
            'time': avg_time,
            'fps': fps,
            'detections': len(boxes)
        })

        print(f"{cfg['label']}: {avg_time:.2f}ms ({fps:.1f} FPS), 检测数: {len(boxes)}")

    # 可视化
    labels = [r['label'] for r in results]
    times = [r['time'] for r in results]
    fps_values = [r['fps'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 处理时间
    bars1 = axes[0].bar(labels, times, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('配置')
    axes[0].set_ylabel('处理时间 (ms)')
    axes[0].set_title('HOG检测处理时间对比')
    for bar, t in zip(bars1, times):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{t:.1f}ms', ha='center', va='bottom')

    # FPS
    bars2 = axes[1].bar(labels, fps_values, color='seagreen', edgecolor='black')
    axes[1].set_xlabel('配置')
    axes[1].set_ylabel('帧率 (FPS)')
    axes[1].set_title('HOG检测帧率对比')
    for bar, f in zip(bars2, fps_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{f:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('hog_performance.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results

results = benchmark_hog_detection()
```

### 6.2 代码示例11：GPU加速HOG（使用CUDA）

```python
import cv2
import numpy as np
import time

def check_cuda_support():
    """检查CUDA支持"""
    print("OpenCV版本:", cv2.__version__)
    print("CUDA设备数量:", cv2.cuda.getCudaEnabledDeviceCount())

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA支持: 可用")
        return True
    else:
        print("CUDA支持: 不可用")
        return False

# 检查CUDA
has_cuda = check_cuda_support()

# 创建测试图像
img = np.random.randint(50, 200, (720, 1280), dtype=np.uint8)
cv2.rectangle(img, (200, 200), (400, 500), 220, -1)
cv2.rectangle(img, (800, 300), (1100, 600), 180, -1)

# CPU HOG检测
hog_cpu = cv2.HOGDescriptor()
hog_cpu.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 测试CPU性能
cpu_times = []
for _ in range(10):
    start = time.time()
    boxes, _ = hog_cpu.detectMultiScale(img, winStride=(8, 8), scale=1.05)
    cpu_times.append((time.time() - start) * 1000)

cpu_avg = np.mean(cpu_times)
print(f"\nCPU HOG检测: {cpu_avg:.2f}ms ({1000/cpu_avg:.1f} FPS)")

# 如果有CUDA支持，测试GPU性能
if has_cuda:
    try:
        # 上传图像到GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        # GPU HOG检测器
        hog_gpu = cv2.cuda.HOG_create()
        hog_gpu.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # 预热
        _ = hog_gpu.detectMultiScale(gpu_img)

        # 测试GPU性能
        gpu_times = []
        for _ in range(10):
            start = time.time()
            boxes = hog_gpu.detectMultiScale(gpu_img)
            gpu_times.append((time.time() - start) * 1000)

        gpu_avg = np.mean(gpu_times)
        speedup = cpu_avg / gpu_avg

        print(f"GPU HOG检测: {gpu_avg:.2f}ms ({1000/gpu_avg:.1f} FPS)")
        print(f"加速比: {speedup:.2f}x")

    except Exception as e:
        print(f"GPU测试失败: {e}")
else:
    print("\n提示: 安装支持CUDA的OpenCV版本可获得GPU加速")
    print("可使用 pip install opencv-contrib-python 安装")
```

## 7. HOG应用场景

### 7.1 常见应用

| 应用领域 | 描述 | 典型场景 |
|----------|------|----------|
| 行人检测 | 识别图像中的行人 | 自动驾驶、监控 |
| 车辆检测 | 检测汽车、卡车等 | 交通监控 |
| 人脸检测 | 辅助人脸定位 | 安防系统 |
| 手势识别 | 识别手部姿态 | 人机交互 |
| 物体识别 | 识别特定物体 | 工业检测 |

### 7.2 HOG的优缺点

**优点：**
- 对光照变化鲁棒（使用梯度）
- 计算相对简单
- 对几何变形有一定容忍度
- 适合描述物体整体形状

**缺点：**
- 没有旋转不变性
- 尺度不变性需要多尺度扫描
- 特征维度较高
- 速度不如深度学习方法快

## 8. 本章小结

哼，笨蛋，HOG特征本小姐可是讲得够详细了吧！(￣▽￣)

### 核心要点回顾

1. **HOG计算流程**
   - 梯度计算 → Cell直方图 → Block归一化 → 特征拼接

2. **关键参数**
   - Cell Size：通常8×8像素
   - Block Size：通常2×2 cells
   - 方向bins：通常9个（0-180°）

3. **应用方式**
   - 使用预训练检测器（如行人检测）
   - 训练自定义检测器（HOG+SVM）

4. **性能优化**
   - 调整winStride和scale平衡精度和速度
   - 使用GPU加速
   - 应用NMS减少重叠检测

## 9. 课后练习

1. **基础练习**：手动实现一个简化版的HOG特征提取器，不使用OpenCV的HOGDescriptor。

2. **进阶练习**：使用HOG+SVM训练一个自定义物体检测器，能够检测特定形状的目标。

3. **综合练习**：在视频流中实现实时行人检测，并计算平均处理帧率。

4. **挑战练习**：比较HOG和深度学习方法（如YOLO）在行人检测上的性能差异。

---

哼，HOG虽然是比较老的算法了，但在某些场景下还是很有用的！笨蛋你可要好好学，本小姐才不是担心你学不会呢！(,,>﹏<,,)
