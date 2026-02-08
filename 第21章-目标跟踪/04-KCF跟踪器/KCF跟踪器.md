# KCF跟踪器

> 哼，颜色直方图跟踪太简单了！本小姐哈雷酱这就来教你KCF跟踪器，这可是基于相关滤波的高性能跟踪算法，速度快精度高，专业人士都在用呢！(￣▽￣)／

## 一、KCF算法原理

### 1.1 什么是KCF

**KCF（Kernelized Correlation Filters）** 是核化相关滤波跟踪器：

- **核心思想**：学习一个相关滤波器来区分目标和背景
- **创新点**：利用循环矩阵和FFT加速计算
- **特点**：速度快、精度高、在线学习

### 1.2 算法优势

| 特性 | 说明 |
|------|------|
| **速度** | 利用FFT实现高效计算 |
| **精度** | 使用HOG特征提高判别力 |
| **鲁棒性** | 在线更新模型 |
| **实时性** | 可达数百FPS |

### 1.3 核心概念

1. **循环矩阵**：通过循环移位生成训练样本
2. **相关滤波**：在频域进行高效相关运算
3. **核技巧**：非线性映射提高判别力
4. **HOG特征**：梯度方向直方图特征

## 二、代码示例

### 示例1：基础KCF跟踪

```python
"""
示例1：基础KCF跟踪
本小姐来演示KCF的强大性能！
"""
import cv2
import time

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("无法读取视频！")
    exit()

# 选择跟踪目标
print("请用鼠标选择跟踪目标...")
bbox = cv2.selectROI('Select Target', frame, False, True)
cv2.destroyWindow('Select Target')

# 创建KCF跟踪器
tracker = cv2.legacy.TrackerKCF_create()

# 初始化跟踪器
tracker.init(frame, bbox)

# 性能统计
frame_count = 0
total_time = 0

print("KCF跟踪已启动")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 计时
    start_time = time.time()

    # 更新跟踪器
    success, bbox = tracker.update(frame)

    elapsed = (time.time() - start_time) * 1000
    total_time += elapsed
    frame_count += 1

    if success:
        # 绘制边界框
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'KCF Tracking', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Tracking Failed!', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示性能信息
    fps = 1000 / elapsed if elapsed > 0 else 0
    avg_fps = 1000 / (total_time / frame_count) if frame_count > 0 else 0

    cv2.putText(frame, f'FPS: {fps:.1f} (avg: {avg_fps:.1f})', (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('KCF Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n平均FPS: {avg_fps:.2f}")
cap.release()
cv2.destroyAllWindows()
```

### 示例2：KCF参数配置

```python
"""
示例2：KCF参数配置
自定义KCF跟踪器参数！
"""
import cv2

def create_custom_kcf():
    """创建自定义参数的KCF跟踪器"""
    # 获取KCF参数
    params = cv2.legacy.TrackerKCF_Params()

    # 检测区域缩放
    params.detect_thresh = 0.5  # 检测阈值

    # 特征设置
    params.sigma = 0.2  # 高斯核带宽
    params.lambda_val = 0.0001  # 正则化参数
    params.interp_factor = 0.075  # 模型更新率
    params.output_sigma_factor = 0.1  # 输出响应高斯带宽

    # 特征类型
    params.compressed_size = 2  # 压缩后的特征维度
    params.desc_npca = 0  # PCA维度
    params.desc_pca = 0

    # 创建跟踪器
    tracker = cv2.legacy.TrackerKCF_create(params)

    return tracker

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

print("选择跟踪目标...")
bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

# 使用自定义参数
tracker = create_custom_kcf()
tracker.init(frame, bbox)

print("自定义KCF跟踪")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Custom KCF', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例3：KCF与其他跟踪器对比

```python
"""
示例3：KCF与其他跟踪器对比
对比不同跟踪器的性能！
"""
import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

print("选择跟踪目标...")
bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

# 创建多个跟踪器
trackers = {
    'KCF': cv2.legacy.TrackerKCF_create(),
    'MOSSE': cv2.legacy.TrackerMOSSE_create(),
    'CSRT': cv2.legacy.TrackerCSRT_create(),
}

# 初始化所有跟踪器
for name, tracker in trackers.items():
    tracker.init(frame, bbox)

# 性能记录
performance = {name: {'times': [], 'success': 0, 'total': 0}
               for name in trackers}

colors = {
    'KCF': (0, 255, 0),
    'MOSSE': (255, 0, 0),
    'CSRT': (0, 0, 255),
}

print("跟踪器性能对比")
print("按 'q' 退出查看统计")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = frame.copy()

    for name, tracker in trackers.items():
        start_time = time.time()
        success, box = tracker.update(frame)
        elapsed = (time.time() - start_time) * 1000

        performance[name]['times'].append(elapsed)
        performance[name]['total'] += 1
        if success:
            performance[name]['success'] += 1

            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(result, (x, y), (x + w, y + h), colors[name], 2)

    # 显示性能
    y_offset = 25
    for name, perf in performance.items():
        avg_time = np.mean(perf['times']) if perf['times'] else 0
        fps = 1000 / avg_time if avg_time > 0 else 0
        success_rate = perf['success'] / perf['total'] * 100 if perf['total'] > 0 else 0

        text = f"{name}: {fps:.0f}fps, {success_rate:.0f}%"
        cv2.putText(result, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)
        y_offset += 25

    cv2.imshow('Tracker Comparison', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 显示统计
print("\n=== 性能统计 ===")
for name, perf in performance.items():
    avg_time = np.mean(perf['times']) if perf['times'] else 0
    fps = 1000 / avg_time if avg_time > 0 else 0
    success_rate = perf['success'] / perf['total'] * 100 if perf['total'] > 0 else 0
    print(f"{name}: {fps:.1f}fps, 成功率: {success_rate:.1f}%")

cap.release()
cv2.destroyAllWindows()
```

### 示例4：带重新初始化的KCF

```python
"""
示例4：带重新初始化的KCF
跟踪失败时自动重新初始化！
"""
import cv2
import numpy as np

class ReinitKCFTracker:
    def __init__(self):
        self.tracker = None
        self.bbox = None
        self.template = None
        self.lost_count = 0
        self.max_lost = 30

    def init(self, frame, bbox):
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.template = self._extract_template(frame, bbox)
        self.lost_count = 0

    def _extract_template(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        return frame[y:y+h, x:x+w].copy()

    def update(self, frame):
        if self.tracker is None:
            return False, None

        success, bbox = self.tracker.update(frame)

        if success:
            self.bbox = bbox
            self.lost_count = 0
            return True, bbox
        else:
            self.lost_count += 1

            if self.lost_count < self.max_lost:
                # 尝试模板匹配恢复
                recovered = self._try_recover(frame)
                if recovered is not None:
                    self.init(frame, recovered)
                    return True, recovered

            return False, self.bbox

    def _try_recover(self, frame):
        if self.template is None or self.bbox is None:
            return None

        x, y, w, h = [int(v) for v in self.bbox]
        margin = 50

        # 搜索区域
        sx = max(0, x - margin)
        sy = max(0, y - margin)
        ex = min(frame.shape[1], x + w + margin)
        ey = min(frame.shape[0], y + h + margin)

        search = frame[sy:ey, sx:ex]

        if search.shape[0] < self.template.shape[0] or \
           search.shape[1] < self.template.shape[1]:
            return None

        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.5:
            new_x = sx + max_loc[0]
            new_y = sy + max_loc[1]
            return (new_x, new_y, self.template.shape[1], self.template.shape[0])

        return None

# 使用
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    exit()

bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

tracker = ReinitKCFTracker()
tracker.init(frame, bbox)

print("KCF自动重新初始化")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox = tracker.update(frame)

    result = frame.copy()

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, 'Tracking', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(result, f'Lost ({tracker.lost_count})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Reinit KCF', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例5：KCF多目标跟踪

```python
"""
示例5：KCF多目标跟踪
同时跟踪多个目标！
"""
import cv2
import numpy as np

class MultiKCFTracker:
    def __init__(self):
        self.trackers = []
        self.next_id = 0

    def add_target(self, frame, bbox):
        tracker = cv2.legacy.TrackerKCF_create()
        tracker.init(frame, bbox)

        color = tuple(map(int, np.random.randint(0, 255, 3)))

        self.trackers.append({
            'id': self.next_id,
            'tracker': tracker,
            'bbox': bbox,
            'color': color,
            'active': True
        })
        self.next_id += 1
        return self.next_id - 1

    def update(self, frame):
        results = []

        for item in self.trackers:
            if item['active']:
                success, bbox = item['tracker'].update(frame)
                item['bbox'] = bbox
                item['active'] = success

                results.append({
                    'id': item['id'],
                    'bbox': bbox,
                    'active': success,
                    'color': item['color']
                })

        return results

    def remove_inactive(self):
        self.trackers = [t for t in self.trackers if t['active']]

# 使用
cap = cv2.VideoCapture(0)
multi_tracker = MultiKCFTracker()

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
                tid = multi_tracker.add_target(frame, (x1, y1, w, h))
                print(f"添加目标 ID:{tid}")

cv2.namedWindow('Multi KCF')
cv2.setMouseCallback('Multi KCF', mouse_callback)

print("多目标KCF跟踪")
print("用鼠标框选添加目标")
print("按 'c' 清除失效目标")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = multi_tracker.update(frame)

    result = frame.copy()

    for r in results:
        if r['active']:
            x, y, w, h = [int(v) for v in r['bbox']]
            cv2.rectangle(result, (x, y), (x + w, y + h), r['color'], 2)
            cv2.putText(result, f"ID:{r['id']}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, r['color'], 2)

    active_count = sum(1 for r in results if r['active'])
    cv2.putText(result, f'Active: {active_count}/{len(results)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 绘制选择框
    if selecting and start_point:
        cv2.rectangle(result, start_point,
                      (cv2.getWindowImageRect('Multi KCF')[2],
                       cv2.getWindowImageRect('Multi KCF')[3]),
                      (0, 255, 255), 2)

    cv2.imshow('Multi KCF', result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        multi_tracker.remove_inactive()
        print("清除失效目标")

cap.release()
cv2.destroyAllWindows()
```

### 示例6：KCF跟踪置信度

```python
"""
示例6：KCF跟踪置信度
估计跟踪结果的置信度！
"""
import cv2
import numpy as np
from collections import deque

class ConfidenceKCFTracker:
    def __init__(self):
        self.tracker = None
        self.prev_bbox = None
        self.template = None
        self.confidence_history = deque(maxlen=30)

    def init(self, frame, bbox):
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.prev_bbox = bbox

        x, y, w, h = [int(v) for v in bbox]
        self.template = frame[y:y+h, x:x+w].copy()

    def update(self, frame):
        if self.tracker is None:
            return False, None, 0

        success, bbox = self.tracker.update(frame)

        if not success:
            return False, self.prev_bbox, 0

        # 计算置信度
        confidence = self._compute_confidence(frame, bbox)
        self.confidence_history.append(confidence)

        self.prev_bbox = bbox

        return True, bbox, confidence

    def _compute_confidence(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]

        # 边界检查
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0

        current = frame[y:y+h, x:x+w]

        if current.shape[0] < 10 or current.shape[1] < 10:
            return 0

        # 模板匹配置信度
        try:
            current_resized = cv2.resize(current, (self.template.shape[1], self.template.shape[0]))
            result = cv2.matchTemplate(current_resized, self.template, cv2.TM_CCOEFF_NORMED)
            confidence = float(result[0][0])
        except:
            confidence = 0

        # 运动一致性
        if self.prev_bbox is not None:
            dx = abs(bbox[0] - self.prev_bbox[0])
            dy = abs(bbox[1] - self.prev_bbox[1])
            motion = np.sqrt(dx**2 + dy**2)
            motion_conf = max(0, 1 - motion / 50)
            confidence = confidence * 0.7 + motion_conf * 0.3

        return max(0, min(1, confidence))

    def get_avg_confidence(self):
        return np.mean(self.confidence_history) if self.confidence_history else 0

# 使用
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

tracker = ConfidenceKCFTracker()
tracker.init(frame, bbox)

print("KCF置信度估计")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox, confidence = tracker.update(frame)

    result = frame.copy()

    if success:
        x, y, w, h = [int(v) for v in bbox]

        # 根据置信度选择颜色
        if confidence > 0.6:
            color = (0, 255, 0)
        elif confidence > 0.3:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

    # 显示置信度
    avg_conf = tracker.get_avg_confidence()
    cv2.putText(result, f'Confidence: {confidence:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result, f'Avg Confidence: {avg_conf:.2f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 置信度条
    bar_width = int(confidence * 200)
    cv2.rectangle(result, (10, 80), (210, 100), (100, 100, 100), -1)
    cv2.rectangle(result, (10, 80), (10 + bar_width, 100), (0, 255, 0), -1)

    cv2.imshow('Confidence KCF', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例7：KCF与检测器结合

```python
"""
示例7：KCF与检测器结合
使用人脸检测初始化KCF跟踪！
"""
import cv2

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

tracker = None
tracking = False

print("KCF + 人脸检测")
print("按 'd' 检测并跟踪人脸")
print("按 'r' 重新检测")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = frame.copy()

    if not tracking:
        # 检测模式
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(result, f'Detected: {len(faces)} faces', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, "Press 'd' to track", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    else:
        # 跟踪模式
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, 'KCF Tracking', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(result, 'Lost - Press r to redetect', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('KCF + Detection', result)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d') and not tracking:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            tracker = cv2.legacy.TrackerKCF_create()
            tracker.init(frame, (x, y, w, h))
            tracking = True
            print("开始跟踪人脸")
    elif key == ord('r'):
        tracking = False
        tracker = None
        print("重新检测")

cap.release()
cv2.destroyAllWindows()
```

### 示例8：KCF尺度估计

```python
"""
示例8：KCF尺度估计
结合多尺度模板匹配估计目标尺度！
"""
import cv2
import numpy as np

class ScaleKCFTracker:
    def __init__(self):
        self.tracker = None
        self.template = None
        self.scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        self.current_scale = 1.0
        self.original_size = None

    def init(self, frame, bbox):
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.tracker.init(frame, bbox)

        x, y, w, h = [int(v) for v in bbox]
        self.template = frame[y:y+h, x:x+w].copy()
        self.original_size = (w, h)
        self.current_scale = 1.0

    def update(self, frame):
        if self.tracker is None:
            return False, None, 1.0

        success, bbox = self.tracker.update(frame)

        if not success:
            return False, None, self.current_scale

        # 估计尺度
        x, y, w, h = [int(v) for v in bbox]
        best_scale = self._estimate_scale(frame, (x, y, w, h))

        # 平滑更新
        self.current_scale = 0.8 * self.current_scale + 0.2 * best_scale

        # 调整边界框
        new_w = int(self.original_size[0] * self.current_scale)
        new_h = int(self.original_size[1] * self.current_scale)
        new_x = x + (w - new_w) // 2
        new_y = y + (h - new_h) // 2

        adjusted_bbox = (new_x, new_y, new_w, new_h)

        return True, adjusted_bbox, self.current_scale

    def _estimate_scale(self, frame, bbox):
        x, y, w, h = bbox
        best_score = -1
        best_scale = 1.0

        for scale in self.scales:
            # 调整模板大小
            new_w = int(self.template.shape[1] * scale)
            new_h = int(self.template.shape[0] * scale)

            if new_w < 10 or new_h < 10:
                continue

            scaled_template = cv2.resize(self.template, (new_w, new_h))

            # 搜索区域
            margin = 20
            sx = max(0, x - margin)
            sy = max(0, y - margin)
            ex = min(frame.shape[1], x + w + margin)
            ey = min(frame.shape[0], y + h + margin)

            search = frame[sy:ey, sx:ex]

            if search.shape[0] < scaled_template.shape[0] or \
               search.shape[1] < scaled_template.shape[1]:
                continue

            result = cv2.matchTemplate(search, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_scale = scale

        return best_scale

# 使用
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

tracker = ScaleKCFTracker()
tracker.init(frame, bbox)

print("KCF尺度估计")
print("按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox, scale = tracker.update(frame)

    result = frame.copy()

    if success and bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(result, f'Scale: {scale:.2f}x', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f'Size: {w}x{h}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Scale KCF', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例9：KCF性能分析

```python
"""
示例9：KCF性能分析
详细分析KCF的性能指标！
"""
import cv2
import numpy as np
import time
from collections import deque

class KCFPerformanceAnalyzer:
    def __init__(self):
        self.tracker = None
        self.frame_times = deque(maxlen=100)
        self.positions = deque(maxlen=100)
        self.success_count = 0
        self.total_count = 0

    def init(self, frame, bbox):
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.positions.append((bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2))

    def update(self, frame):
        if self.tracker is None:
            return False, None

        start_time = time.time()
        success, bbox = self.tracker.update(frame)
        elapsed = (time.time() - start_time) * 1000

        self.frame_times.append(elapsed)
        self.total_count += 1

        if success:
            self.success_count += 1
            self.positions.append((bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2))

        return success, bbox

    def get_stats(self):
        avg_time = np.mean(self.frame_times) if self.frame_times else 0
        fps = 1000 / avg_time if avg_time > 0 else 0
        success_rate = self.success_count / self.total_count * 100 if self.total_count > 0 else 0

        # 计算轨迹稳定性
        if len(self.positions) > 1:
            velocities = []
            for i in range(1, len(self.positions)):
                dx = self.positions[i][0] - self.positions[i-1][0]
                dy = self.positions[i][1] - self.positions[i-1][1]
                velocities.append(np.sqrt(dx**2 + dy**2))
            stability = np.std(velocities) if velocities else 0
        else:
            stability = 0

        return {
            'avg_time_ms': avg_time,
            'fps': fps,
            'success_rate': success_rate,
            'total_frames': self.total_count,
            'stability': stability
        }

# 使用
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
bbox = cv2.selectROI('Select', frame, False, True)
cv2.destroyWindow('Select')

analyzer = KCFPerformanceAnalyzer()
analyzer.init(frame, bbox)

print("KCF性能分析")
print("按 'q' 退出查看统计")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox = analyzer.update(frame)

    result = frame.copy()

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    stats = analyzer.get_stats()

    # 显示统计
    cv2.putText(result, f"FPS: {stats['fps']:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result, f"Time: {stats['avg_time_ms']:.1f}ms", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result, f"Success: {stats['success_rate']:.1f}%", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(result, f"Stability: {stats['stability']:.2f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('KCF Performance', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 最终统计
stats = analyzer.get_stats()
print("\n=== 最终统计 ===")
print(f"平均FPS: {stats['fps']:.2f}")
print(f"平均处理时间: {stats['avg_time_ms']:.2f}ms")
print(f"成功率: {stats['success_rate']:.2f}%")
print(f"总帧数: {stats['total_frames']}")

cap.release()
cv2.destroyAllWindows()
```

### 示例10：完整KCF跟踪系统

```python
"""
示例10：完整KCF跟踪系统
集成所有功能的完整系统！
"""
import cv2
import numpy as np
from collections import deque
import time

class CompleteKCFSystem:
    def __init__(self):
        self.tracker = None
        self.initialized = False
        self.trajectory = deque(maxlen=100)
        self.frame_times = deque(maxlen=30)
        self.confidence_history = deque(maxlen=30)
        self.template = None
        self.lost_count = 0

    def init(self, frame, bbox):
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.tracker.init(frame, bbox)

        x, y, w, h = [int(v) for v in bbox]
        self.template = frame[y:y+h, x:x+w].copy()

        self.trajectory.clear()
        self.trajectory.append((x + w//2, y + h//2))

        self.initialized = True
        self.lost_count = 0

    def update(self, frame):
        if not self.initialized:
            return False, None, 0

        start_time = time.time()
        success, bbox = self.tracker.update(frame)
        elapsed = (time.time() - start_time) * 1000
        self.frame_times.append(elapsed)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            self.trajectory.append((x + w//2, y + h//2))

            # 计算置信度
            confidence = self._compute_confidence(frame, bbox)
            self.confidence_history.append(confidence)

            self.lost_count = 0
            return True, bbox, confidence
        else:
            self.lost_count += 1

            # 尝试恢复
            if self.lost_count < 30:
                recovered = self._try_recover(frame)
                if recovered is not None:
                    self.init(frame, recovered)
                    return True, recovered, 0.5

            return False, None, 0

    def _compute_confidence(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]

        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0

        try:
            current = frame[y:y+h, x:x+w]
            current_resized = cv2.resize(current, (self.template.shape[1], self.template.shape[0]))
            result = cv2.matchTemplate(current_resized, self.template, cv2.TM_CCOEFF_NORMED)
            return float(result[0][0])
        except:
            return 0

    def _try_recover(self, frame):
        if self.template is None or not self.trajectory:
            return None

        last_pos = self.trajectory[-1]
        margin = 50

        sx = max(0, last_pos[0] - margin - self.template.shape[1]//2)
        sy = max(0, last_pos[1] - margin - self.template.shape[0]//2)
        ex = min(frame.shape[1], last_pos[0] + margin + self.template.shape[1]//2)
        ey = min(frame.shape[0], last_pos[1] + margin + self.template.shape[0]//2)

        search = frame[sy:ey, sx:ex]

        if search.shape[0] < self.template.shape[0] or \
           search.shape[1] < self.template.shape[1]:
            return None

        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.5:
            return (sx + max_loc[0], sy + max_loc[1],
                    self.template.shape[1], self.template.shape[0])
        return None

    def get_fps(self):
        if not self.frame_times:
            return 0
        return 1000 / np.mean(self.frame_times)

    def get_avg_confidence(self):
        return np.mean(self.confidence_history) if self.confidence_history else 0

    def draw(self, frame, bbox, confidence):
        result = frame.copy()

        # 绘制轨迹
        for i in range(1, len(self.trajectory)):
            alpha = i / len(self.trajectory)
            color = (int(255 * (1-alpha)), int(255 * alpha), 0)
            thickness = max(1, int(3 * alpha))
            cv2.line(result, self.trajectory[i-1], self.trajectory[i], color, thickness)

        # 绘制边界框
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]

            if confidence > 0.5:
                box_color = (0, 255, 0)
            elif confidence > 0.3:
                box_color = (0, 255, 255)
            else:
                box_color = (0, 0, 255)

            cv2.rectangle(result, (x, y), (x + w, y + h), box_color, 2)

        return result

def main():
    cap = cv2.VideoCapture(0)
    system = CompleteKCFSystem()

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

    cv2.namedWindow('Complete KCF')
    cv2.setMouseCallback('Complete KCF', mouse_callback)

    print("完整KCF跟踪系统")
    print("用鼠标框选目标")
    print("按 'r' 重置")
    print("按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 初始化
        if current_bbox is not None:
            system.init(frame, current_bbox)
            current_bbox = None

        result = frame.copy()

        # 更新
        if system.initialized:
            success, bbox, confidence = system.update(frame)
            result = system.draw(result, bbox, confidence)

            # 显示统计
            fps = system.get_fps()
            avg_conf = system.get_avg_confidence()

            cv2.putText(result, f'FPS: {fps:.1f}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f'Confidence: {confidence:.2f}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f'Avg Conf: {avg_conf:.2f}', (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if not success:
                cv2.putText(result, f'Lost ({system.lost_count})', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(result, 'Draw box to start', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 选择框
        if selecting and start_point:
            cv2.rectangle(result, start_point,
                          (cv2.getWindowImageRect('Complete KCF')[2],
                           cv2.getWindowImageRect('Complete KCF')[3]),
                          (0, 255, 255), 2)

        cv2.imshow('Complete KCF', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            system = CompleteKCFSystem()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

## 三、KCF优缺点

### 优点
- 速度极快（可达数百FPS）
- 精度较高
- 在线学习
- 实现相对简单

### 缺点
- 不能处理尺度变化
- 快速运动可能失败
- 边界效应问题
- 遮挡恢复能力有限

## 四、总结

哼，KCF的精髓本小姐已经全部教给你了！(￣ω￣)

**核心要点：**
1. 利用循环矩阵和FFT加速
2. 使用HOG特征提高判别力
3. 在线更新保持适应性
4. 速度快适合实时应用
5. 需结合其他方法处理尺度变化
6. 可与检测器配合使用

才、才不是担心笨蛋学不会才讲这么细的！(,,>.<,,)

## 五、练习题

1. **基础练习**：实现基础KCF跟踪
2. **进阶练习**：添加置信度估计
3. **挑战练习**：实现多目标KCF跟踪
4. **综合练习**：结合检测器创建完整系统

---

*本小姐哈雷酱的KCF教程到此结束！下一节教你其他跟踪器！* (￣▽￣*)
