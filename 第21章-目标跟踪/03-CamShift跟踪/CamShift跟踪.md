# CamShift跟踪

> 哼，MeanShift不能处理尺度变化是个大问题！本小姐哈雷酱这就来教你CamShift，它可是MeanShift的升级版，能自动调整窗口大小呢！(￣▽￣)／

## 一、CamShift算法原理

### 1.1 什么是CamShift

**CamShift（Continuously Adaptive Mean Shift）** 是MeanShift的自适应版本：

- **核心改进**：自动调整搜索窗口的大小和方向
- **名称来源**：持续自适应均值漂移
- **优势**：能处理目标尺度变化和旋转

### 1.2 与MeanShift的区别

| 特性 | MeanShift | CamShift |
|------|-----------|----------|
| 窗口大小 | 固定 | 自适应 |
| 窗口方向 | 固定 | 可旋转 |
| 尺度变化 | 不能处理 | 能处理 |
| 输出 | 矩形 | 旋转矩形 |

### 1.3 算法步骤

1. **初始化**：与MeanShift相同
2. **MeanShift迭代**：先执行标准MeanShift
3. **计算二阶矩**：根据收敛位置计算目标的二阶矩
4. **调整窗口**：根据二阶矩调整窗口大小和方向
5. **返回结果**：返回旋转矩形

## 二、代码示例

### 示例1：基础CamShift跟踪

```python
"""
示例1：基础CamShift跟踪
本小姐来演示CamShift的自适应特性！
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("无法读取视频！")
    exit()

# 选择初始跟踪区域
print("请用鼠标选择跟踪目标...")
bbox = cv2.selectROI('Select Target', frame, False, True)
cv2.destroyWindow('Select Target')

x, y, w, h = [int(v) for v in bbox]
track_window = (x, y, w, h)

# 提取ROI并计算直方图
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 终止条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

print("CamShift跟踪已启动")
print("注意观察窗口大小的自适应变化")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # 应用CamShift
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # 绘制旋转矩形
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # 显示窗口信息
    center, size, angle = ret
    cv2.putText(frame, f'Size: {int(size[0])}x{int(size[1])}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Angle: {angle:.1f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('CamShift Tracking', frame)
    cv2.imshow('Back Projection', dst)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例2：对比MeanShift和CamShift

```python
"""
示例2：对比MeanShift和CamShift
直观展示两者的区别！
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

# 两个跟踪窗口
ms_window = (x, y, w, h)
cs_window = (x, y, w, h)

# 计算直方图
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

print("MeanShift vs CamShift 对比")
print("红色=MeanShift，绿色=CamShift")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # MeanShift
    ret_ms, ms_window = cv2.meanShift(dst, ms_window, term_crit)
    x, y, w, h = ms_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(frame, 'MeanShift', (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # CamShift
    ret_cs, cs_window = cv2.CamShift(dst, cs_window, term_crit)
    pts = cv2.boxPoints(ret_cs)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    center, size, angle = ret_cs
    cv2.putText(frame, 'CamShift', (int(center[0]), int(center[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 信息对比
    cv2.putText(frame, f'MS: {w}x{h}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f'CS: {int(size[0])}x{int(size[1])}, {angle:.0f}deg',
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Comparison', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例3：旋转目标跟踪

```python
"""
示例3：旋转目标跟踪
CamShift可以跟踪旋转的目标！
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

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 记录角度历史
angle_history = []

print("旋转目标跟踪")
print("尝试旋转目标观察效果")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    center, size, angle = ret
    angle_history.append(angle)
    if len(angle_history) > 30:
        angle_history.pop(0)

    # 绘制旋转矩形
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # 绘制中心点和方向
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # 绘制方向箭头
    length = max(size) / 2
    end_x = int(cx + length * np.cos(np.radians(angle)))
    end_y = int(cy + length * np.sin(np.radians(angle)))
    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (255, 0, 0), 2)

    # 显示信息
    cv2.putText(frame, f'Angle: {angle:.1f} deg', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Size: {int(size[0])}x{int(size[1])}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 角度变化趋势
    if len(angle_history) > 1:
        angle_change = angle_history[-1] - angle_history[-2]
        direction = "Rotating CW" if angle_change > 1 else ("Rotating CCW" if angle_change < -1 else "Stable")
        cv2.putText(frame, direction, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow('Rotation Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例4：尺度变化跟踪

```python
"""
示例4：尺度变化跟踪
演示CamShift的尺度自适应能力！
"""
import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

x, y, w, h = [int(v) for v in bbox]
track_window = (x, y, w, h)
initial_size = (w, h)

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 尺度历史
scale_history = deque(maxlen=50)

print("尺度变化跟踪")
print("移动目标靠近或远离摄像头")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    center, size, angle = ret

    # 计算相对尺度
    current_area = size[0] * size[1]
    initial_area = initial_size[0] * initial_size[1]
    scale = np.sqrt(current_area / initial_area) if initial_area > 0 else 1
    scale_history.append(scale)

    # 绘制旋转矩形
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # 显示尺度信息
    cv2.putText(frame, f'Scale: {scale:.2f}x', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Size: {int(size[0])}x{int(size[1])}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 尺度趋势判断
    if len(scale_history) > 10:
        recent_scales = list(scale_history)[-10:]
        if recent_scales[-1] > recent_scales[0] * 1.1:
            trend = "Approaching"
            color = (0, 255, 0)
        elif recent_scales[-1] < recent_scales[0] * 0.9:
            trend = "Moving Away"
            color = (0, 0, 255)
        else:
            trend = "Stable"
            color = (255, 255, 0)

        cv2.putText(frame, trend, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 绘制尺度条
    bar_width = int(scale * 100)
    bar_width = min(300, max(10, bar_width))
    cv2.rectangle(frame, (10, 110), (10 + bar_width, 130), (0, 255, 0), -1)
    cv2.rectangle(frame, (10, 110), (310, 130), (255, 255, 255), 1)
    cv2.line(frame, (110, 108), (110, 132), (0, 0, 255), 2)  # 1x标记

    cv2.imshow('Scale Tracking', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例5：带轨迹的CamShift

```python
"""
示例5：带轨迹的CamShift
绘制目标的运动轨迹！
"""
import cv2
import numpy as np
from collections import deque

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

# 轨迹
trajectory = deque(maxlen=100)
# 尺寸历史（用于绘制变化的轨迹宽度）
size_history = deque(maxlen=100)

print("CamShift轨迹可视化")
print("按 'c' 清除轨迹")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret_val, track_window = cv2.CamShift(dst, track_window, term_crit)

    center, size, angle = ret_val

    # 记录轨迹
    trajectory.append((int(center[0]), int(center[1])))
    size_history.append(max(size) / 50)  # 归一化尺寸

    # 绘制渐变轨迹
    for i in range(1, len(trajectory)):
        if trajectory[i-1] is None or trajectory[i] is None:
            continue

        alpha = i / len(trajectory)
        # 颜色从蓝渐变到绿
        color = (int(255 * (1-alpha)), int(255 * alpha), 0)
        # 线宽根据尺寸变化
        thickness = max(1, int(size_history[i] if i < len(size_history) else 2))
        cv2.line(frame, trajectory[i-1], trajectory[i], color, thickness)

    # 绘制旋转矩形
    pts = cv2.boxPoints(ret_val)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    # 绘制中心点
    cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

    cv2.putText(frame, f'Points: {len(trajectory)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Trajectory CamShift', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        trajectory.clear()
        size_history.clear()

cap.release()
cv2.destroyAllWindows()
```

### 示例6：多通道CamShift

```python
"""
示例6：多通道CamShift
使用H和S两个通道提高精度！
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

# 2D直方图（H和S）
roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

print("2D直方图CamShift")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2D反向投影
    dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # 模糊处理减少噪声
    dst = cv2.GaussianBlur(dst, (5, 5), 0)

    # CamShift
    ret_val, track_window = cv2.CamShift(dst, track_window, term_crit)

    # 绘制结果
    pts = cv2.boxPoints(ret_val)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    center, size, angle = ret_val
    cv2.putText(frame, '2D Histogram CamShift', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Size: {int(size[0])}x{int(size[1])}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('2D CamShift', frame)
    cv2.imshow('Back Projection', dst)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例7：自适应模型更新

```python
"""
示例7：自适应模型更新
根据跟踪质量动态更新模型！
"""
import cv2
import numpy as np

class AdaptiveCamShift:
    def __init__(self, frame, bbox, update_rate=0.1):
        x, y, w, h = [int(v) for v in bbox]
        self.track_window = (x, y, w, h)
        self.update_rate = update_rate

        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))

        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.original_hist = self.roi_hist.copy()
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def update(self, frame, adaptive=True):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        ret_val, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)

        center, size, angle = ret_val

        # 计算置信度
        x, y, w, h = self.track_window
        if w > 0 and h > 0:
            roi = dst[y:y+h, x:x+w]
            confidence = np.mean(roi) / 255 if roi.size > 0 else 0
        else:
            confidence = 0

        # 自适应更新（仅在置信度高时）
        if adaptive and confidence > 0.4 and size[0] > 10 and size[1] > 10:
            x, y, w, h = self.track_window
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                roi = frame[y:min(y+h, frame.shape[0]), x:min(x+w, frame.shape[1])]
                if roi.size > 0:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
                    new_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                    cv2.normalize(new_hist, new_hist, 0, 255, cv2.NORM_MINMAX)

                    self.roi_hist = (1 - self.update_rate) * self.roi_hist + \
                                    self.update_rate * new_hist

        return ret_val, confidence

    def reset_model(self):
        self.roi_hist = self.original_hist.copy()

# 使用演示
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

tracker = AdaptiveCamShift(frame, bbox)
adaptive_enabled = True

print("自适应CamShift")
print("按 'a' 切换自适应更新")
print("按 'r' 重置模型")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ret_val, confidence = tracker.update(frame, adaptive=adaptive_enabled)

    # 绘制
    pts = cv2.boxPoints(ret_val)
    pts = np.int0(pts)

    color = (0, 255, 0) if confidence > 0.4 else (0, 0, 255)
    cv2.polylines(frame, [pts], True, color, 2)

    cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f'Adaptive: {"ON" if adaptive_enabled else "OFF"}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Adaptive CamShift', frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        adaptive_enabled = not adaptive_enabled
    elif key == ord('r'):
        tracker.reset_model()

cap.release()
cv2.destroyAllWindows()
```

### 示例8：多目标CamShift

```python
"""
示例8：多目标CamShift
同时跟踪多个目标！
"""
import cv2
import numpy as np

class CamShiftTarget:
    def __init__(self, frame, bbox, target_id):
        x, y, w, h = [int(v) for v in bbox]
        self.track_window = (x, y, w, h)
        self.id = target_id

        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.color = tuple(map(int, np.random.randint(0, 255, 3)))
        self.active = True
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def update(self, hsv_frame):
        if not self.active:
            return None

        dst = cv2.calcBackProject([hsv_frame], [0], self.roi_hist, [0, 180], 1)
        ret_val, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)

        # 检查有效性
        center, size, angle = ret_val
        if size[0] < 5 or size[1] < 5:
            self.active = False
            return None

        return ret_val

targets = []
next_id = 0
selecting = False
start_point = None

def mouse_callback(event, x, y, flags, param):
    global selecting, start_point, next_id

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
                target = CamShiftTarget(frame, (x1, y1, w, h), next_id)
                targets.append(target)
                next_id += 1
                print(f"添加目标 ID:{target.id}")

cap = cv2.VideoCapture(0)
cv2.namedWindow('Multi CamShift')
cv2.setMouseCallback('Multi CamShift', mouse_callback)

print("多目标CamShift")
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
    for target in targets:
        if target.active:
            ret_val = target.update(hsv)

            if ret_val is not None:
                pts = cv2.boxPoints(ret_val)
                pts = np.int0(pts)
                cv2.polylines(result, [pts], True, target.color, 2)

                center = ret_val[0]
                cv2.putText(result, f'ID:{target.id}', (int(center[0]), int(center[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, target.color, 2)

    # 移除失效目标
    targets = [t for t in targets if t.active]

    # 绘制选择框
    if selecting and start_point:
        cv2.rectangle(result, start_point,
                      (cv2.getWindowImageRect('Multi CamShift')[2],
                       cv2.getWindowImageRect('Multi CamShift')[3]),
                      (0, 255, 255), 2)

    cv2.putText(result, f'Targets: {len(targets)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Multi CamShift', result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        targets.clear()
        print("清除所有目标")

cap.release()
cv2.destroyAllWindows()
```

### 示例9：CamShift人脸跟踪

```python
"""
示例9：CamShift人脸跟踪
结合人脸检测初始化CamShift！
"""
import cv2
import numpy as np

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

tracker = None
track_window = None
roi_hist = None
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

print("CamShift人脸跟踪")
print("按 'd' 检测人脸并开始跟踪")
print("按 'r' 重新检测")
print("按 'q' 退出")

tracking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = frame.copy()

    if not tracking:
        # 检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(result, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, "Press 'd' to start tracking", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    else:
        # CamShift跟踪
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret_val, track_window = cv2.CamShift(dst, track_window, term_crit)

        # 绘制旋转矩形
        pts = cv2.boxPoints(ret_val)
        pts = np.int0(pts)
        cv2.polylines(result, [pts], True, (0, 255, 0), 2)

        center, size, angle = ret_val
        cv2.putText(result, 'Tracking Face', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Face CamShift', result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d') and not tracking:
        # 检测并选择第一个人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            track_window = (x, y, w, h)

            # 计算直方图
            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            tracking = True
            print("开始跟踪人脸")
    elif key == ord('r'):
        tracking = False
        print("重新检测")

cap.release()
cv2.destroyAllWindows()
```

### 示例10：完整CamShift系统

```python
"""
示例10：完整CamShift系统
集成所有功能的完整跟踪系统！
"""
import cv2
import numpy as np
from collections import deque

class CompleteCamShiftTracker:
    def __init__(self):
        self.track_window = None
        self.roi_hist = None
        self.original_hist = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.trajectory = deque(maxlen=100)
        self.size_history = deque(maxlen=30)
        self.angle_history = deque(maxlen=30)
        self.initialized = False
        self.adaptive = True
        self.update_rate = 0.05

    def init(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        self.track_window = (x, y, w, h)
        self.initial_size = (w, h)

        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))

        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.original_hist = self.roi_hist.copy()

        self.trajectory.clear()
        self.trajectory.append((x + w//2, y + h//2))
        self.initialized = True

    def update(self, frame):
        if not self.initialized:
            return None, 0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        ret_val, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)

        center, size, angle = ret_val

        # 记录历史
        self.trajectory.append((int(center[0]), int(center[1])))
        self.size_history.append(size)
        self.angle_history.append(angle)

        # 计算置信度
        x, y, w, h = self.track_window
        roi = dst[y:y+h, x:x+w]
        confidence = np.mean(roi) / 255 if roi.size > 0 else 0

        # 自适应更新
        if self.adaptive and confidence > 0.4:
            self._update_model(frame)

        return ret_val, confidence

    def _update_model(self, frame):
        x, y, w, h = self.track_window
        if w > 5 and h > 5:
            y_end = min(y + h, frame.shape[0])
            x_end = min(x + w, frame.shape[1])
            roi = frame[max(0,y):y_end, max(0,x):x_end]

            if roi.size > 0:
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array([0, 60, 32]), np.array([180, 255, 255]))
                new_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(new_hist, new_hist, 0, 255, cv2.NORM_MINMAX)

                self.roi_hist = (1 - self.update_rate) * self.roi_hist + \
                                self.update_rate * new_hist

    def get_scale(self):
        if not self.size_history:
            return 1.0
        current = self.size_history[-1]
        initial = self.initial_size
        return np.sqrt((current[0] * current[1]) / (initial[0] * initial[1] + 1e-6))

    def draw(self, frame, ret_val, confidence):
        result = frame.copy()

        if ret_val is None:
            return result

        # 绘制轨迹
        for i in range(1, len(self.trajectory)):
            alpha = i / len(self.trajectory)
            color = (int(255 * (1-alpha)), int(255 * alpha), 0)
            thickness = max(1, int(3 * alpha))
            cv2.line(result, self.trajectory[i-1], self.trajectory[i], color, thickness)

        # 绘制旋转矩形
        pts = cv2.boxPoints(ret_val)
        pts = np.int0(pts)
        box_color = (0, 255, 0) if confidence > 0.4 else (0, 0, 255)
        cv2.polylines(result, [pts], True, box_color, 2)

        # 中心点
        center = ret_val[0]
        cv2.circle(result, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

        return result

def main():
    cap = cv2.VideoCapture(0)
    tracker = CompleteCamShiftTracker()

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

    cv2.namedWindow('Complete CamShift')
    cv2.setMouseCallback('Complete CamShift', mouse_callback)

    print("完整CamShift跟踪系统")
    print("用鼠标框选目标")
    print("按 'a' 切换自适应")
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

        result = frame.copy()

        # 跟踪
        if tracker.initialized:
            ret_val, confidence = tracker.update(frame)
            result = tracker.draw(result, ret_val, confidence)

            # 显示信息
            center, size, angle = ret_val
            scale = tracker.get_scale()

            cv2.putText(result, f'Confidence: {confidence:.2f}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f'Size: {int(size[0])}x{int(size[1])}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f'Angle: {angle:.1f}', (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f'Scale: {scale:.2f}x', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f'Adaptive: {"ON" if tracker.adaptive else "OFF"}',
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(result, 'Draw box to start', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 绘制选择框
        if selecting and start_point:
            cv2.rectangle(result, start_point,
                          (cv2.getWindowImageRect('Complete CamShift')[2],
                           cv2.getWindowImageRect('Complete CamShift')[3]),
                          (0, 255, 255), 2)

        cv2.imshow('Complete CamShift', result)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            tracker.adaptive = not tracker.adaptive
        elif key == ord('r'):
            tracker = CompleteCamShiftTracker()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

## 三、CamShift优缺点

### 优点
- 自适应窗口大小
- 可以跟踪旋转目标
- 计算速度快
- 实现简单

### 缺点
- 仍然依赖颜色特征
- 相似颜色容易干扰
- 快速运动可能失败
- 需要目标与背景有明显颜色差异

## 四、总结

哼，CamShift的所有知识本小姐都教给你了！(￣ω￣)

**核心要点：**
1. CamShift是MeanShift的自适应版本
2. 自动调整窗口大小和方向
3. 输出旋转矩形更精确描述目标
4. 适合跟踪变化的目标
5. 自适应更新提高鲁棒性
6. 仍然基于颜色直方图

才、才不是怕笨蛋理解不了才讲这么细的！(,,>.<,,)

## 五、练习题

1. **基础练习**：实现CamShift跟踪
2. **进阶练习**：可视化尺度和角度变化
3. **挑战练习**：结合人脸检测初始化跟踪
4. **综合练习**：实现带置信度的自适应CamShift

---

*本小姐哈雷酱的CamShift教程到此结束！下一节教你KCF跟踪器！* (￣▽￣*)
