# SURF特征

## 1. SURF算法概述

SURF（Speeded Up Robust Features，加速稳健特征）是由Herbert Bay等人于2006年提出的特征提取算法。SURF是SIFT的加速版本，在保持类似性能的同时大幅提高了计算速度。

### 1.1 SURF的特点

| 特点 | 说明 |
|------|------|
| 速度快 | 比SIFT快3-5倍 |
| 尺度不变 | 使用Hessian矩阵近似 |
| 旋转不变 | 基于Haar小波的方向分配 |
| 可选描述子 | 64维或128维 |
| 专利限制 | 商业使用需授权 |

### 1.2 SURF与SIFT比较

| 特性 | SIFT | SURF |
|------|------|------|
| 描述子维度 | 128 | 64或128 |
| 检测速度 | 慢 | 快 |
| 尺度空间 | DoG | Hessian矩阵 |
| 方向分配 | 梯度直方图 | Haar小波响应 |
| 描述子计算 | 梯度方向直方图 | Haar小波响应 |

### 1.3 SURF算法流程

1. **尺度空间构建**：使用盒式滤波器近似Hessian矩阵
2. **兴趣点定位**：在尺度空间中检测极值点
3. **方向分配**：使用Haar小波计算主方向
4. **描述子生成**：基于Haar小波响应的描述子

**注意**：SURF在OpenCV的主发行版中由于专利限制已被移除。需要安装`opencv-contrib-python`包才能使用。

## 2. SURF原理简介

### 2.1 积分图像

SURF使用积分图像加速计算：

$$II(x, y) = \sum_{i \leq x, j \leq y} I(i, j)$$

任意矩形区域的和可以通过4次查表计算得到。

### 2.2 Hessian矩阵

SURF使用Hessian矩阵的行列式检测特征点：

$$H(x, \sigma) = \begin{bmatrix} L_{xx}(x, \sigma) & L_{xy}(x, \sigma) \\ L_{xy}(x, \sigma) & L_{yy}(x, \sigma) \end{bmatrix}$$

使用盒式滤波器近似高斯二阶导数。

### 2.3 描述子构建

1. 以特征点为中心取20σ×20σ区域
2. 分成4×4子区域
3. 每个子区域计算Haar小波响应
4. 每个子区域产生4维向量：$(\sum d_x, \sum d_y, \sum |d_x|, \sum |d_y|)$
5. 总共4×4×4=64维描述子

## 3. 代码示例

### 示例1：检查SURF可用性

```python
"""
示例1：检查SURF可用性
检查OpenCV是否支持SURF
"""
import cv2
import numpy as np

def check_surf_available():
    """检查SURF是否可用"""
    try:
        surf = cv2.xfeatures2d.SURF_create()
        print("✓ SURF可用")
        return True
    except AttributeError:
        print("✗ SURF不可用")
        print("  SURF需要opencv-contrib-python包")
        print("  安装命令: pip install opencv-contrib-python")
        return False
    except cv2.error as e:
        print(f"✗ SURF错误: {e}")
        return False

# 检查
surf_available = check_surf_available()

# 如果不可用，提供替代方案
if not surf_available:
    print("\n替代方案:")
    print("1. 使用SIFT (免费)")
    print("2. 使用ORB (快速，免费)")
    print("3. 使用AKAZE (免费)")

    # 演示替代方案
    print("\n使用ORB作为替代:")
    orb = cv2.ORB_create()

    # 创建测试图像
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

    kps, descs = orb.detectAndCompute(img, None)
    print(f"ORB检测到 {len(kps)} 个特征点")
```

### 示例2：基本SURF特征提取

```python
"""
示例2：基本SURF特征提取
使用SURF提取特征（需要opencv-contrib）
"""
import cv2
import numpy as np

# 创建测试图像
img = np.zeros((400, 600), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), 200, -1)
cv2.circle(img, (400, 150), 100, 200, -1)
cv2.rectangle(img, (100, 280), (500, 380), 150, -1)

# 添加纹理
for i in range(0, 600, 12):
    cv2.line(img, (i, 0), (i, 400), 100, 1)
for j in range(0, 400, 12):
    cv2.line(img, (0, j), (600, j), 100, 1)

try:
    # 创建SURF检测器
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    # 检测特征点并计算描述子
    keypoints, descriptors = surf.detectAndCompute(img, None)

    print("SURF特征提取结果:")
    print("-" * 50)
    print(f"检测到 {len(keypoints)} 个特征点")
    if descriptors is not None:
        print(f"描述子形状: {descriptors.shape}")
        print(f"描述子类型: {descriptors.dtype}")

    # 可视化
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = cv2.drawKeypoints(result, keypoints, None,
                               color=(0, 255, 0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(result, f"SURF: {len(keypoints)} keypoints", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("SURF Features", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except AttributeError:
    print("SURF不可用，使用SIFT作为替代")

    # 使用SIFT作为替代
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    print(f"SIFT检测到 {len(keypoints)} 个特征点")

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = cv2.drawKeypoints(result, keypoints, None,
                               color=(0, 255, 0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("SIFT Features (SURF alternative)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 示例3：SURF参数调节

```python
"""
示例3：SURF参数调节
分析SURF参数对特征提取的影响
"""
import cv2
import numpy as np

def extract_surf_features(img, hessian_threshold=400, n_octaves=4,
                          n_octave_layers=3, extended=False, upright=False):
    """
    提取SURF特征

    Args:
        img: 输入图像
        hessian_threshold: Hessian阈值
        n_octaves: 金字塔层数
        n_octave_layers: 每层的尺度数
        extended: 是否使用128维描述子
        upright: 是否忽略方向（更快但不旋转不变）

    Returns:
        keypoints, descriptors
    """
    try:
        surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
            extended=extended,
            upright=upright
        )
        return surf.detectAndCompute(img, None)
    except AttributeError:
        # SURF不可用时使用SIFT
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(img, None)

# 创建测试图像
img = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
cv2.circle(img, (280, 100), 60, 200, -1)
cv2.rectangle(img, (50, 200), (350, 280), 150, -1)

# 添加纹理
for i in range(0, 400, 8):
    cv2.line(img, (i, 0), (i, 300), 100, 1)

# 参数测试
param_sets = [
    {"name": "Default", "hessian_threshold": 400, "extended": False, "upright": False},
    {"name": "Low threshold", "hessian_threshold": 100, "extended": False, "upright": False},
    {"name": "High threshold", "hessian_threshold": 1000, "extended": False, "upright": False},
    {"name": "Extended (128D)", "hessian_threshold": 400, "extended": True, "upright": False},
    {"name": "Upright", "hessian_threshold": 400, "extended": False, "upright": True},
    {"name": "Upright+Extended", "hessian_threshold": 400, "extended": True, "upright": True},
]

print("SURF参数影响分析:")
print("=" * 70)
print(f"{'参数设置':>20} {'特征数':>10} {'描述子维度':>15}")
print("-" * 70)

results = []
for params in param_sets:
    kps, descs = extract_surf_features(
        img,
        hessian_threshold=params["hessian_threshold"],
        extended=params["extended"],
        upright=params["upright"]
    )

    desc_dim = descs.shape[1] if descs is not None else 0
    print(f"{params['name']:>20} {len(kps):>10} {desc_dim:>15}")

    results.append((params['name'], kps, descs))

# 可视化
n_results = min(6, len(results))
rows = 2
cols = 3
canvas = np.zeros((rows * 300, cols * 400, 3), dtype=np.uint8)

for idx, (name, kps, descs) in enumerate(results[:n_results]):
    row = idx // cols
    col = idx % cols
    x_off = col * 400
    y_off = row * 300

    vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    vis = cv2.drawKeypoints(vis, kps, None,
                            color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(vis, name, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, f"n={len(kps)}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    canvas[y_off:y_off+300, x_off:x_off+400] = vis

cv2.imshow("SURF Parameter Comparison", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例4：SURF vs SIFT 速度比较

```python
"""
示例4：SURF vs SIFT 速度比较
比较两种算法的性能
"""
import cv2
import numpy as np
import time

def benchmark_feature_extractor(extractor, img, name, iterations=10):
    """测试特征提取性能"""
    times = []
    keypoints = None
    descriptors = None

    for _ in range(iterations):
        start = time.time()
        keypoints, descriptors = extractor.detectAndCompute(img, None)
        times.append(time.time() - start)

    return {
        'name': name,
        'keypoints': len(keypoints) if keypoints else 0,
        'descriptor_dim': descriptors.shape[1] if descriptors is not None else 0,
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000
    }

# 创建测试图像
img = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
for _ in range(10):
    x, y = np.random.randint(50, 550), np.random.randint(50, 400)
    cv2.rectangle(img, (x, y), (x+60, y+60), np.random.randint(100, 250), -1)

# 创建特征提取器
extractors = []

# SIFT
extractors.append(('SIFT', cv2.SIFT_create()))

# 尝试添加SURF
try:
    extractors.append(('SURF-64', cv2.xfeatures2d.SURF_create(extended=False)))
    extractors.append(('SURF-128', cv2.xfeatures2d.SURF_create(extended=True)))
    extractors.append(('SURF-Upright', cv2.xfeatures2d.SURF_create(upright=True)))
except AttributeError:
    print("SURF不可用，跳过SURF测试")

# 添加其他算法作为对比
extractors.append(('ORB', cv2.ORB_create()))
extractors.append(('AKAZE', cv2.AKAZE_create()))

print("特征提取算法性能比较:")
print("=" * 80)
print(f"图像大小: {img.shape[1]}x{img.shape[0]}")
print("-" * 80)
print(f"{'算法':>15} {'特征数':>10} {'维度':>8} {'平均时间(ms)':>15} {'标准差':>10}")
print("-" * 80)

results = []
for name, extractor in extractors:
    result = benchmark_feature_extractor(extractor, img, name)
    results.append(result)
    print(f"{result['name']:>15} {result['keypoints']:>10} {result['descriptor_dim']:>8} "
          f"{result['avg_time_ms']:>15.2f} {result['std_time_ms']:>10.2f}")

print("-" * 80)

# 速度排名
sorted_results = sorted(results, key=lambda x: x['avg_time_ms'])
print("\n按速度排名（从快到慢）:")
for i, r in enumerate(sorted_results, 1):
    print(f"{i}. {r['name']}: {r['avg_time_ms']:.2f}ms")

# 可视化部分结果
vis_extractors = extractors[:min(3, len(extractors))]
canvas = np.zeros((480, 640 * len(vis_extractors), 3), dtype=np.uint8)

for i, (name, extractor) in enumerate(vis_extractors):
    kps, _ = extractor.detectAndCompute(img, None)
    vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    vis = cv2.drawKeypoints(vis, kps, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(vis, f"{name}: {len(kps)} pts", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    canvas[:, i*640:(i+1)*640] = vis

cv2.imshow("Feature Extractors Comparison", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例5：SURF描述子分析

```python
"""
示例5：SURF描述子分析
分析SURF描述子的结构
"""
import cv2
import numpy as np

# 创建简单图像
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 200, -1)

# 添加明显的纹理
for i in range(50, 150, 10):
    cv2.line(img, (i, 50), (i, 150), 100, 2)

try:
    # 64维SURF
    surf_64 = cv2.xfeatures2d.SURF_create(hessianThreshold=100, extended=False)
    kps_64, descs_64 = surf_64.detectAndCompute(img, None)

    # 128维SURF
    surf_128 = cv2.xfeatures2d.SURF_create(hessianThreshold=100, extended=True)
    kps_128, descs_128 = surf_128.detectAndCompute(img, None)

    print("SURF描述子分析:")
    print("=" * 60)

    print("\n64维描述子:")
    if descs_64 is not None and len(descs_64) > 0:
        desc = descs_64[0]
        print(f"  形状: {descs_64.shape}")
        print(f"  第一个描述子统计:")
        print(f"    最小值: {desc.min():.4f}")
        print(f"    最大值: {desc.max():.4f}")
        print(f"    平均值: {desc.mean():.4f}")
        print(f"    L2范数: {np.linalg.norm(desc):.4f}")

        # 64维描述子结构：4x4子区域，每个4维
        print(f"\n  结构说明: 4x4=16子区域，每个4维")
        print(f"  每个子区域: (Σdx, Σdy, Σ|dx|, Σ|dy|)")

    print("\n128维描述子:")
    if descs_128 is not None and len(descs_128) > 0:
        desc = descs_128[0]
        print(f"  形状: {descs_128.shape}")
        print(f"  第一个描述子统计:")
        print(f"    最小值: {desc.min():.4f}")
        print(f"    最大值: {desc.max():.4f}")
        print(f"    平均值: {desc.mean():.4f}")
        print(f"    L2范数: {np.linalg.norm(desc):.4f}")

        print(f"\n  结构说明: 4x4=16子区域，每个8维")
        print(f"  分别统计dy<0和dy≥0时的响应")

except AttributeError:
    print("SURF不可用，使用SIFT进行描述子分析")

    sift = cv2.SIFT_create(nfeatures=5)
    kps, descs = sift.detectAndCompute(img, None)

    print("SIFT描述子分析:")
    print("=" * 60)
    if descs is not None and len(descs) > 0:
        desc = descs[0]
        print(f"形状: {descs.shape}")
        print(f"描述子维度: 128")
        print(f"结构: 4x4子区域，每个8方向梯度直方图")

# 可视化
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

try:
    kps = kps_64 if 'kps_64' in dir() else []
except:
    kps = []

if kps:
    result = cv2.drawKeypoints(result, kps, None,
                               color=(0, 255, 0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Features", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例6：SURF特征匹配

```python
"""
示例6：SURF特征匹配
使用SURF进行图像匹配
"""
import cv2
import numpy as np

def surf_match(img1, img2, hessian_threshold=400, ratio=0.75):
    """
    使用SURF进行特征匹配

    Returns:
        keypoints1, keypoints2, good_matches
    """
    try:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
    except AttributeError:
        # 使用SIFT作为后备
        surf = cv2.SIFT_create()
        print("使用SIFT替代SURF")

    # 提取特征
    kp1, desc1 = surf.detectAndCompute(img1, None)
    kp2, desc2 = surf.detectAndCompute(img2, None)

    if desc1 is None or desc2 is None:
        return kp1, kp2, []

    # FLANN匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)

    # 比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return kp1, kp2, sorted(good_matches, key=lambda x: x.distance)

# 创建测试图像对
img1 = np.zeros((300, 400), dtype=np.uint8)
cv2.rectangle(img1, (80, 60), (200, 180), 200, -1)
cv2.circle(img1, (300, 120), 60, 200, -1)
cv2.rectangle(img1, (100, 220), (350, 280), 150, -1)

# 添加纹理
for i in range(0, 400, 10):
    cv2.line(img1, (i, 0), (i, 300), 100, 1)
for j in range(0, 300, 10):
    cv2.line(img1, (0, j), (400, j), 100, 1)

# img2：旋转和缩放
center = (200, 150)
M = cv2.getRotationMatrix2D(center, 20, 0.9)
img2 = cv2.warpAffine(img1, M, (400, 300))
noise = np.random.normal(0, 8, img2.shape).astype(np.int16)
img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 匹配
kp1, kp2, good_matches = surf_match(img1, img2)

print(f"图像1特征点: {len(kp1)}")
print(f"图像2特征点: {len(kp2)}")
print(f"好的匹配: {len(good_matches)}")

# 可视化
result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:30], None,
                         matchColor=(0, 255, 0),
                         singlePointColor=(255, 0, 0),
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.putText(result, f"Matches: {len(good_matches)}", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("SURF Matching", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例7：SURF用于全景拼接

```python
"""
示例7：SURF用于全景拼接
使用SURF特征进行简单的图像拼接
"""
import cv2
import numpy as np

def create_panorama(img1, img2, feature_extractor='surf'):
    """
    使用特征匹配进行图像拼接

    Args:
        img1: 左图
        img2: 右图
        feature_extractor: 'surf' 或 'sift'

    Returns:
        拼接后的图像
    """
    # 创建特征提取器
    try:
        if feature_extractor == 'surf':
            detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        else:
            detector = cv2.SIFT_create()
    except AttributeError:
        detector = cv2.SIFT_create()
        print("使用SIFT替代SURF")

    # 提取特征
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)

    print(f"图像1: {len(kp1)} 特征点")
    print(f"图像2: {len(kp2)} 特征点")

    # 匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)

    # 比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"好的匹配: {len(good_matches)}")

    if len(good_matches) < 4:
        print("匹配点过少，无法计算单应性")
        return None

    # 计算单应性矩阵
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None:
        print("无法计算单应性矩阵")
        return None

    # 计算输出图像大小
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 变换img2的角点
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2, H)

    # 合并所有角点计算边界
    all_corners = np.concatenate([
        np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
        corners2_transformed
    ])

    x_min = int(min(0, all_corners[:, 0, 0].min()))
    x_max = int(max(w1, all_corners[:, 0, 0].max()))
    y_min = int(min(0, all_corners[:, 0, 1].min()))
    y_max = int(max(h1, all_corners[:, 0, 1].max()))

    # 平移矩阵
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    # 拼接
    output_size = (x_max - x_min, y_max - y_min)
    result = cv2.warpPerspective(img2, translation @ H, output_size)

    # 将img1放入结果
    result[-y_min:-y_min+h1, -x_min:-x_min+w1] = img1

    return result, good_matches, kp1, kp2

# 创建两个有重叠区域的图像
width, height = 400, 300
overlap = 100

# 图像1（左）
img1 = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(img1, (50, 50), (200, 200), 200, -1)
cv2.circle(img1, (320, 150), 60, 180, -1)
# 添加纹理
for i in range(0, width, 15):
    cv2.line(img1, (i, 0), (i, height), 100, 1)
for j in range(0, height, 15):
    cv2.line(img1, (0, j), (width, j), 100, 1)

# 图像2（右）- 模拟平移
img2 = np.zeros((height, width), dtype=np.uint8)
# 复制重叠区域并添加新内容
img2[:, :overlap] = img1[:, width-overlap:]
cv2.rectangle(img2, (150, 80), (350, 250), 160, -1)
cv2.circle(img2, (250, 150), 50, 200, -1)
# 添加纹理
for i in range(0, width, 15):
    cv2.line(img2, (i, 0), (i, height), 100, 1)
for j in range(0, height, 15):
    cv2.line(img2, (0, j), (width, j), 100, 1)

# 添加噪声
noise1 = np.random.normal(0, 5, img1.shape).astype(np.int16)
noise2 = np.random.normal(0, 5, img2.shape).astype(np.int16)
img1 = np.clip(img1.astype(np.int16) + noise1, 0, 255).astype(np.uint8)
img2 = np.clip(img2.astype(np.int16) + noise2, 0, 255).astype(np.uint8)

# 拼接
result = create_panorama(img1, img2)

if result is not None:
    panorama, matches, kp1, kp2 = result

    # 显示结果
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.imshow("Panorama", panorama)

    # 显示匹配
    match_vis = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                                matchColor=(0, 255, 0))
    cv2.imshow("Matches", match_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 示例8：SURF特征的Upright模式

```python
"""
示例8：SURF特征的Upright模式
比较普通SURF和Upright SURF
"""
import cv2
import numpy as np

# 创建测试图像
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (80, 80), (220, 220), 200, -1)
# 添加非对称纹理
cv2.rectangle(img, (100, 100), (140, 200), 100, -1)
cv2.circle(img, (180, 120), 20, 100, -1)

# 添加网格
for i in range(0, 300, 10):
    cv2.line(img, (i, 0), (i, 300), 150, 1)
    cv2.line(img, (0, i), (300, i), 150, 1)

try:
    # 普通SURF（计算方向）
    surf_normal = cv2.xfeatures2d.SURF_create(hessianThreshold=300, upright=False)

    # Upright SURF（不计算方向，更快）
    surf_upright = cv2.xfeatures2d.SURF_create(hessianThreshold=300, upright=True)

    # 提取特征
    kps_normal, descs_normal = surf_normal.detectAndCompute(img, None)
    kps_upright, descs_upright = surf_upright.detectAndCompute(img, None)

    print("SURF模式比较:")
    print("=" * 50)
    print(f"{'模式':>15} {'特征数':>10} {'有方向':>10}")
    print("-" * 50)

    # 检查方向
    has_orientation_normal = any(kp.angle >= 0 for kp in kps_normal)
    has_orientation_upright = any(kp.angle >= 0 for kp in kps_upright)

    print(f"{'Normal':>15} {len(kps_normal):>10} {'是' if has_orientation_normal else '否':>10}")
    print(f"{'Upright':>15} {len(kps_upright):>10} {'是' if has_orientation_upright else '否':>10}")

    # 打印方向信息
    print("\n方向信息示例（前5个特征点）:")
    print("-" * 50)
    print(f"{'':>5} {'Normal角度':>15} {'Upright角度':>15}")

    for i in range(min(5, len(kps_normal), len(kps_upright))):
        print(f"{i:>5} {kps_normal[i].angle:>15.1f} {kps_upright[i].angle:>15.1f}")

    # 可视化
    vis_normal = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    vis_upright = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    vis_normal = cv2.drawKeypoints(vis_normal, kps_normal, None,
                                   color=(0, 255, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_upright = cv2.drawKeypoints(vis_upright, kps_upright, None,
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(vis_normal, "Normal SURF", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_upright, "Upright SURF", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    combined = np.hstack([vis_normal, vis_upright])
    cv2.imshow("Normal vs Upright SURF", combined)

except AttributeError:
    print("SURF不可用")
    cv2.imshow("Test Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 4. SURF应用场景

### 4.1 适用场景

| 场景 | 推荐设置 |
|------|----------|
| 实时应用 | Upright=True, 64维 |
| 高精度匹配 | Upright=False, 128维 |
| 移动端 | 考虑使用ORB替代 |
| 物体识别 | 标准设置 |

### 4.2 替代方案

由于SURF有专利限制，以下是推荐的替代方案：

| 替代方案 | 优势 | 劣势 |
|----------|------|------|
| SIFT | 免费，精确 | 较慢 |
| ORB | 免费，快速 | 精度稍低 |
| AKAZE | 免费，多尺度 | 中等速度 |
| BRISK | 免费，多尺度 | 特征较少 |

## 5. 总结

本节介绍了SURF特征：

| 内容 | 要点 |
|------|------|
| 核心原理 | 基于Hessian矩阵和积分图像 |
| 描述子维度 | 64维或128维 |
| 速度 | 比SIFT快3-5倍 |
| 专利状态 | 有专利限制 |
| OpenCV使用 | 需要opencv-contrib-python |

## 6. 练习题

1. **基础练习**：
   - 比较SURF的64维和128维描述子的匹配效果
   - 测试Upright模式对旋转图像的影响

2. **进阶练习**：
   - 实现SURF和SIFT的性能对比测试
   - 使用SURF进行简单的物体识别

3. **实践项目**：
   - 如果SURF不可用，使用其他算法实现相同功能
   - 创建一个支持多种特征提取器的统一接口
