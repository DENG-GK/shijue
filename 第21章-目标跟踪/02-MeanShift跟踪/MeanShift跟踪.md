# MeanShift跟踪

> 哼，MeanShift可是经典的目标跟踪算法！本小姐哈雷酱这就来教你这个基于颜色直方图的优雅算法，笨蛋给我认真学！(￣▽￣)／

## 一、MeanShift算法原理

### 1.1 基本概念

**MeanShift（均值漂移）** 是一种非参数密度估计算法：

- **核心思想**：向数据密度增加最大的方向迭代移动
- **在跟踪中**：寻找颜色分布最相似的区域
- **特点**：简单、快速、无需训练

### 1.2 算法步骤

1. **建立目标模型**：计算目标区域的颜色直方图
2. **在新帧中搜索**：以上一位置为起点
3. **计算权重**：基于直方图反向投影
4. **移动窗口**：向权重质心移动
5. **迭代**：直到收敛或达到最大迭代次数

### 1.3 数学表达

质心计算：
$$x_c = \frac{\sum_{i} x_i w_i}{\sum_{i} w_i}, \quad y_c = \frac{\sum_{i} y_i w_i}{\sum_{i} w_i}$$

其中 $w_i$ 是像素 $(x_i, y_i)$ 的权重（来自反向投影）。

## 二、代码示例

### 示例1：基础MeanShift跟踪

```python
"""
示例1：基础MeanShift跟踪
本小姐来演示最基础的MeanShift跟踪！
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 读取第一帧
ret, frame = cap.read()
if not ret:
    print("无法读取视频！")
    exit()

# 选择初始跟踪区域
print("请用鼠标选择跟踪目标...")
bbox = cv2.selectROI('Select Target', frame, False, True)
cv2.destroyWindow('Select Target')

x, y, w, h = [int(v) for v in bbox]

# 设置初始跟踪窗口
track_window = (x, y, w, h)

# 提取目标区域的ROI
roi = frame[y:y+h, x:x+w]

# 转换到HSV颜色空间
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 创建掩码，过滤低饱和度区域（排除白色和黑色）
mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))

# 计算直方图
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# 归一化直方图
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置MeanShift终止条件
# (类型, 最大迭代次数, 精度)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

print("MeanShift跟踪已启动")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换到HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 计算反向投影
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 应用MeanShift
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # 绘制跟踪结果
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, 'MeanShift', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示反向投影
    cv2.imshow('Back Projection', dst)
    cv2.imshow('MeanShift Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例2：理解反向投影

```python
"""
示例2：理解反向投影
反向投影是MeanShift的核心，本小姐来详细解释！
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

# 选择目标
bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

x, y, w, h = [int(v) for v in bbox]
roi = frame[y:y+h, x:x+w]

# 转换到HSV
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 不同通道的直方图
channels = ['H (色调)', 'S (饱和度)', 'V (亮度)']
hists = []

for i, name in enumerate(channels):
    hist = cv2.calcHist([hsv_roi], [i], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hists.append(hist)

print("反向投影演示")
print("按 'h/s/v' 切换通道")
print("按 'q' 退出")

channel = 0  # 默认H通道
ranges = [[0, 180], [0, 256], [0, 256]]
sizes = [180, 256, 256]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 计算反向投影
    dst = cv2.calcBackProject([hsv], [channel], hists[channel],
                               ranges[channel], 1)

    # 可视化
    result = frame.copy()
    cv2.putText(result, f'Channel: {channels[channel]}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 反向投影热图
    dst_color = cv2.applyColorMap(dst, cv2.COLORMAP_JET)

    cv2.imshow('Original', result)
    cv2.imshow('Back Projection', dst)
    cv2.imshow('Heatmap', dst_color)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        channel = 0
    elif key == ord('s'):
        channel = 1
    elif key == ord('v'):
        channel = 2

cap.release()
cv2.destroyAllWindows()
```

### 示例3：多通道直方图

```python
"""
示例3：多通道直方图
使用H和S两个通道提高跟踪精度！
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

print("选择跟踪目标...")
bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

x, y, w, h = [int(v) for v in bbox]
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 创建掩码
mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))

# 计算2D直方图（H和S通道）
roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

print("2D直方图MeanShift跟踪")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2D反向投影
    dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # 应用高斯模糊减少噪声
    dst = cv2.GaussianBlur(dst, (5, 5), 0)

    # MeanShift
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, '2D Histogram', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Tracking', frame)
    cv2.imshow('Back Projection', dst)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例4：迭代过程可视化

```python
"""
示例4：迭代过程可视化
展示MeanShift的迭代收敛过程！
"""
import cv2
import numpy as np

def meanshift_step_by_step(prob_map, window, max_iter=10):
    """
    逐步执行MeanShift，返回每步的窗口位置
    """
    x, y, w, h = window
    steps = [(x, y)]

    for i in range(max_iter):
        # 提取窗口区域
        roi = prob_map[y:y+h, x:x+w]

        if roi.size == 0:
            break

        # 计算质心
        M = cv2.moments(roi)
        if M['m00'] == 0:
            break

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 计算新位置
        new_x = x + cx - w // 2
        new_y = y + cy - h // 2

        # 边界检查
        new_x = max(0, min(new_x, prob_map.shape[1] - w))
        new_y = max(0, min(new_y, prob_map.shape[0] - h))

        # 检查是否收敛
        if abs(new_x - x) < 1 and abs(new_y - y) < 1:
            break

        x, y = new_x, new_y
        steps.append((x, y))

    return (x, y, w, h), steps

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

x, y, w, h = [int(v) for v in bbox]
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

print("MeanShift迭代可视化")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 逐步执行MeanShift
    new_window, steps = meanshift_step_by_step(dst, track_window)
    track_window = new_window

    result = frame.copy()

    # 绘制迭代路径
    for i, (sx, sy) in enumerate(steps):
        alpha = (i + 1) / len(steps)
        color = (0, int(255 * alpha), int(255 * (1 - alpha)))
        cv2.rectangle(result, (sx, sy), (sx + w, sy + h), color, 1)

    # 绘制最终位置
    x, y, w, h = track_window
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(result, f'Iterations: {len(steps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Iteration Visualization', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例5：自适应直方图更新

```python
"""
示例5：自适应直方图更新
动态更新目标模型以适应外观变化！
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

x, y, w, h = [int(v) for v in bbox]
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 保存原始直方图
original_hist = roi_hist.copy()

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 更新参数
update_rate = 0.1  # 直方图更新率
update_enabled = True

print("自适应MeanShift跟踪")
print("按 'u' 切换自适应更新")
print("按 'r' 重置直方图")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    x, y, w, h = track_window

    # 自适应更新直方图
    if update_enabled and ret:
        # 提取当前跟踪区域
        current_roi = frame[y:y+h, x:x+w]
        if current_roi.size > 0:
            current_hsv = cv2.cvtColor(current_roi, cv2.COLOR_BGR2HSV)
            current_mask = cv2.inRange(current_hsv,
                                       np.array([0, 60, 32]),
                                       np.array([180, 255, 255]))
            current_hist = cv2.calcHist([current_hsv], [0], current_mask,
                                        [180], [0, 180])
            cv2.normalize(current_hist, current_hist, 0, 255, cv2.NORM_MINMAX)

            # 加权更新
            roi_hist = (1 - update_rate) * roi_hist + update_rate * current_hist

    result = frame.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    status = "Adaptive ON" if update_enabled else "Adaptive OFF"
    cv2.putText(result, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Adaptive MeanShift', result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u'):
        update_enabled = not update_enabled
        print(f"自适应更新: {'开启' if update_enabled else '关闭'}")
    elif key == ord('r'):
        roi_hist = original_hist.copy()
        print("直方图已重置")

cap.release()
cv2.destroyAllWindows()
```

### 示例6：带置信度的跟踪

```python
"""
示例6：带置信度的跟踪
计算跟踪置信度，判断跟踪质量！
"""
import cv2
import numpy as np

def compute_tracking_confidence(prob_map, window):
    """计算跟踪置信度"""
    x, y, w, h = window
    roi = prob_map[y:y+h, x:x+w]

    if roi.size == 0:
        return 0

    # 计算窗口内的平均概率
    mean_prob = np.mean(roi)

    # 计算峰值比
    max_prob = np.max(roi)

    # 综合置信度
    confidence = (mean_prob / 255) * 0.5 + (max_prob / 255) * 0.5

    return confidence

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

x, y, w, h = [int(v) for v in bbox]
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

CONFIDENCE_THRESHOLD = 0.3

print("带置信度的MeanShift跟踪")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # 计算置信度
    confidence = compute_tracking_confidence(dst, track_window)

    x, y, w, h = track_window

    result = frame.copy()

    # 根据置信度选择颜色
    if confidence > CONFIDENCE_THRESHOLD:
        color = (0, 255, 0)  # 绿色 - 可信
        status = "Tracking"
    else:
        color = (0, 0, 255)  # 红色 - 不可信
        status = "Uncertain"

    cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
    cv2.putText(result, f'{status} ({confidence:.2f})', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 置信度条
    bar_width = int(confidence * 200)
    cv2.rectangle(result, (10, 30), (210, 50), (100, 100, 100), -1)
    cv2.rectangle(result, (10, 30), (10 + bar_width, 50), color, -1)
    cv2.putText(result, f'Confidence: {confidence:.2f}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Confidence Tracking', result)
    cv2.imshow('Back Projection', dst)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例7：多目标MeanShift

```python
"""
示例7：多目标MeanShift
同时用MeanShift跟踪多个目标！
"""
import cv2
import numpy as np

class MeanShiftTarget:
    def __init__(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        self.track_window = (x, y, w, h)

        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.color = tuple(map(int, np.random.randint(0, 255, 3)))
        self.active = True

    def update(self, hsv_frame, term_crit):
        if not self.active:
            return False

        dst = cv2.calcBackProject([hsv_frame], [0], self.roi_hist, [0, 180], 1)
        ret, self.track_window = cv2.meanShift(dst, self.track_window, term_crit)

        return ret > 0

targets = []
selecting = False
start_point = None

def mouse_callback(event, x, y, flags, param):
    global selecting, start_point

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and selecting:
        selecting = False
        x1, y1 = start_point
        w, h = x - x1, y - y1

        if w > 10 and h > 10:
            ret, frame = cap.read()
            if ret:
                target = MeanShiftTarget(frame, (x1, y1, w, h))
                targets.append(target)
                print(f"添加目标，当前共 {len(targets)} 个")

cap = cv2.VideoCapture(0)
cv2.namedWindow('Multi MeanShift')
cv2.setMouseCallback('Multi MeanShift', mouse_callback)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

print("多目标MeanShift跟踪")
print("用鼠标框选添加目标")
print("按 'c' 清除所有目标")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    result = frame.copy()

    # 更新所有目标
    for i, target in enumerate(targets):
        if target.active:
            target.update(hsv, term_crit)

            x, y, w, h = target.track_window
            cv2.rectangle(result, (x, y), (x + w, y + h), target.color, 2)
            cv2.putText(result, f'T{i}', (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, target.color, 2)

    # 绘制选择框
    if selecting and start_point:
        cv2.rectangle(result, start_point,
                      (cv2.getWindowImageRect('Multi MeanShift')[2],
                       cv2.getWindowImageRect('Multi MeanShift')[3]),
                      (0, 255, 255), 2)

    cv2.putText(result, f'Targets: {len(targets)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Multi MeanShift', result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        targets.clear()
        print("清除所有目标")

cap.release()
cv2.destroyAllWindows()
```

### 示例8：MeanShift参数调优

```python
"""
示例8：MeanShift参数调优
调节参数找到最佳效果！
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow('Settings')

def nothing(x):
    pass

# 创建滑动条
cv2.createTrackbar('H_min', 'Settings', 0, 180, nothing)
cv2.createTrackbar('S_min', 'Settings', 60, 255, nothing)
cv2.createTrackbar('V_min', 'Settings', 32, 255, nothing)
cv2.createTrackbar('Max_iter', 'Settings', 10, 50, nothing)
cv2.createTrackbar('Epsilon', 'Settings', 1, 10, nothing)

ret, frame = cap.read()
if not ret:
    exit()

print("选择跟踪目标...")
bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

x, y, w, h = [int(v) for v in bbox]
track_window = (x, y, w, h)

print("MeanShift参数调优")
print("调节滑动条优化跟踪效果")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 获取参数
    h_min = cv2.getTrackbarPos('H_min', 'Settings')
    s_min = cv2.getTrackbarPos('S_min', 'Settings')
    v_min = cv2.getTrackbarPos('V_min', 'Settings')
    max_iter = max(1, cv2.getTrackbarPos('Max_iter', 'Settings'))
    epsilon = max(0.1, cv2.getTrackbarPos('Epsilon', 'Settings'))

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 使用当前参数计算直方图
    x, y, w, h = track_window
    if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
        roi = frame[max(0,y):min(frame.shape[0],y+h), max(0,x):min(frame.shape[1],x+w)]
        if roi.size > 0:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi,
                               np.array([h_min, s_min, v_min]),
                               np.array([180, 255, 255]))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    result = frame.copy()
    x, y, w, h = track_window
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示参数
    info = [
        f'H_min: {h_min}',
        f'S_min: {s_min}',
        f'V_min: {v_min}',
        f'Max_iter: {max_iter}',
        f'Epsilon: {epsilon}'
    ]

    for i, text in enumerate(info):
        cv2.putText(result, text, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Settings', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例9：MeanShift与模板匹配结合

```python
"""
示例9：MeanShift与模板匹配结合
结合两种方法提高鲁棒性！
"""
import cv2
import numpy as np

class HybridTracker:
    def __init__(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        self.track_window = (x, y, w, h)

        # MeanShift模型
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        # 模板
        self.template = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # MeanShift跟踪
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        ret, ms_window = cv2.meanShift(dst, self.track_window, self.term_crit)

        # 模板匹配验证
        x, y, w, h = ms_window

        # 在MeanShift结果附近搜索
        search_margin = 20
        sx = max(0, x - search_margin)
        sy = max(0, y - search_margin)
        ex = min(gray.shape[1], x + w + search_margin)
        ey = min(gray.shape[0], y + h + search_margin)

        search_region = gray[sy:ey, sx:ex]

        if search_region.shape[0] >= self.template.shape[0] and \
           search_region.shape[1] >= self.template.shape[1]:
            result = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.5:
                # 使用模板匹配结果
                tm_x = sx + max_loc[0]
                tm_y = sy + max_loc[1]
                self.track_window = (tm_x, tm_y, w, h)
                return True, self.track_window, max_val, 'Template'
            else:
                # 使用MeanShift结果
                self.track_window = ms_window
                return True, self.track_window, max_val, 'MeanShift'
        else:
            self.track_window = ms_window
            return True, self.track_window, 0, 'MeanShift'

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

tracker = HybridTracker(frame, bbox)

print("混合跟踪器")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox, confidence, method = tracker.update(frame)

    result = frame.copy()

    if success:
        x, y, w, h = [int(v) for v in bbox]
        color = (0, 255, 0) if method == 'Template' else (255, 255, 0)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        cv2.putText(result, f'Method: {method}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(result, f'Confidence: {confidence:.2f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Hybrid Tracker', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例10：完整MeanShift跟踪系统

```python
"""
示例10：完整MeanShift跟踪系统
集成所有功能的完整系统！
"""
import cv2
import numpy as np
from collections import deque

class CompleteMeanShiftTracker:
    def __init__(self):
        self.track_window = None
        self.roi_hist = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.trajectory = deque(maxlen=50)
        self.confidence_history = deque(maxlen=30)
        self.initialized = False
        self.adaptive_update = True
        self.update_rate = 0.05

    def init(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        self.track_window = (x, y, w, h)

        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.trajectory.clear()
        self.trajectory.append((x + w//2, y + h//2))
        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            return False, None, 0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)

        x, y, w, h = self.track_window

        # 计算置信度
        roi = dst[y:y+h, x:x+w]
        confidence = np.mean(roi) / 255 if roi.size > 0 else 0
        self.confidence_history.append(confidence)

        # 更新轨迹
        self.trajectory.append((x + w//2, y + h//2))

        # 自适应更新
        if self.adaptive_update and confidence > 0.3:
            current_roi = frame[y:y+h, x:x+w]
            if current_roi.size > 0:
                current_hsv = cv2.cvtColor(current_roi, cv2.COLOR_BGR2HSV)
                current_mask = cv2.inRange(current_hsv,
                                           np.array([0, 60, 32]),
                                           np.array([180, 255, 255]))
                current_hist = cv2.calcHist([current_hsv], [0], current_mask,
                                            [180], [0, 180])
                cv2.normalize(current_hist, current_hist, 0, 255, cv2.NORM_MINMAX)
                self.roi_hist = (1 - self.update_rate) * self.roi_hist + \
                                self.update_rate * current_hist

        return True, self.track_window, confidence

    def draw(self, frame):
        result = frame.copy()

        if not self.initialized:
            return result

        x, y, w, h = self.track_window

        # 绘制轨迹
        for i in range(1, len(self.trajectory)):
            alpha = i / len(self.trajectory)
            thickness = max(1, int(3 * alpha))
            color = (int(255 * (1-alpha)), int(255 * alpha), 0)
            cv2.line(result, self.trajectory[i-1], self.trajectory[i], color, thickness)

        # 绘制边界框
        avg_conf = np.mean(self.confidence_history) if self.confidence_history else 0
        if avg_conf > 0.3:
            color = (0, 255, 0)
        elif avg_conf > 0.15:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        return result

def main():
    cap = cv2.VideoCapture(0)
    tracker = CompleteMeanShiftTracker()

    selecting = False
    start_point = None
    current_bbox = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_point, current_bbox

        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and selecting:
            selecting = False
            x1, y1 = start_point
            w, h = x - x1, y - y1
            if w > 10 and h > 10:
                current_bbox = (x1, y1, w, h)

    cv2.namedWindow('Complete MeanShift')
    cv2.setMouseCallback('Complete MeanShift', mouse_callback)

    print("完整MeanShift跟踪系统")
    print("用鼠标框选目标")
    print("按 'a' 切换自适应更新")
    print("按 'r' 重置")
    print("按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 初始化
        if current_bbox is not None:
            tracker.init(frame, current_bbox)
            current_bbox = None
            print("开始跟踪")

        # 更新
        if tracker.initialized:
            success, bbox, confidence = tracker.update(frame)

        # 绘制
        result = tracker.draw(frame)

        # 显示信息
        if tracker.initialized:
            avg_conf = np.mean(tracker.confidence_history) if tracker.confidence_history else 0
            cv2.putText(result, f'Confidence: {avg_conf:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f'Adaptive: {"ON" if tracker.adaptive_update else "OFF"}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(result, 'Draw box to start', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 绘制选择框
        if selecting and start_point:
            cv2.rectangle(result, start_point,
                          (cv2.getWindowImageRect('Complete MeanShift')[2],
                           cv2.getWindowImageRect('Complete MeanShift')[3]),
                          (0, 255, 255), 2)

        cv2.imshow('Complete MeanShift', result)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            tracker.adaptive_update = not tracker.adaptive_update
            print(f"自适应更新: {'开启' if tracker.adaptive_update else '关闭'}")
        elif key == ord('r'):
            tracker = CompleteMeanShiftTracker()
            print("重置跟踪器")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

## 三、MeanShift优缺点

### 优点
- 算法简单，易于实现
- 计算速度快
- 对部分遮挡有一定鲁棒性
- 无需训练

### 缺点
- 窗口大小固定，无法处理尺度变化
- 对快速运动效果差
- 需要目标颜色与背景有明显差异
- 容易跟踪到相似颜色区域

## 四、实际应用场景

1. **人脸跟踪**：肤色区域跟踪
2. **手势跟踪**：手部区域跟踪
3. **物体跟踪**：颜色鲜艳的物体
4. **简单监控**：固定摄像头场景

## 五、总结

哼，MeanShift的精髓本小姐已经全部教给你了！(￣ω￣)

**核心要点：**
1. 基于颜色直方图匹配
2. 反向投影是核心步骤
3. 迭代移动窗口到质心
4. 自适应更新提高鲁棒性
5. 置信度判断跟踪质量
6. 窗口大小固定是主要缺陷

才、才不是怕笨蛋学不会才讲这么细的！(,,>.<,,)

## 六、练习题

1. **基础练习**：实现基础的MeanShift跟踪
2. **进阶练习**：添加自适应直方图更新
3. **挑战练习**：实现多目标MeanShift跟踪
4. **综合练习**：结合模板匹配提高鲁棒性

---

*本小姐哈雷酱的MeanShift教程到此结束！下一节教你CamShift！* (￣▽￣*)
