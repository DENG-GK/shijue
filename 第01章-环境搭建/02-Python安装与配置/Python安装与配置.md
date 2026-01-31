# Python 安装与配置

> Python 是一门简单易学、功能强大的编程语言，是计算机视觉和机器学习的首选语言之一。

---

## 📖 理论部分

### 1. Python 简介

**Python** 是一种解释型、面向对象、动态类型的高级编程语言。

**Python 的优势：**
- ✅ **语法简洁**：易学易用，代码可读性强
- ✅ **生态丰富**：NumPy、OpenCV、TensorFlow 等强大库
- ✅ **跨平台**：Windows、macOS、Linux 通用
- ✅ **社区活跃**：大量教程和开源项目
- ✅ **应用广泛**：Web开发、数据科学、AI、自动化等

**为什么选择 Python 做计算机视觉？**
- OpenCV 官方支持 Python
- NumPy 提供高效的数组计算
- 深度学习框架（PyTorch、TensorFlow）都有 Python 接口
- Jupyter Notebook 支持交互式开发

---

### 2. Python 版本选择

**Python 主要版本：**
- **Python 2.x**：已于 2020 年停止维护，不推荐使用
- **Python 3.x**：当前主流版本

**推荐版本：Python 3.8+**
- **Python 3.8**：稳定、兼容性好
- **Python 3.9**：性能提升
- **Python 3.10**：新增模式匹配
- **Python 3.11**：性能大幅提升（推荐）
- **Python 3.12**：最新版，部分库可能不兼容

**本教程推荐：Python 3.11.x**

---

### 3. 下载与安装

#### Windows 系统安装

**步骤1：下载 Python**

访问官网：https://www.python.org/downloads/

或直接下载（64位）：
- Python 3.11.x：https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe

**步骤2：安装 Python**

1. 双击安装包 `python-3.11.7-amd64.exe`
2. **⚠️ 重要：勾选 "Add Python 3.11 to PATH"**（必须勾选！）
3. 选择 "Customize installation"（自定义安装）
4. **Optional Features**（可选功能）：
   - ✅ Documentation（文档）
   - ✅ pip（包管理器，必选）
   - ✅ tcl/tk and IDLE（图形界面）
   - ✅ Python test suite
   - ✅ py launcher（Python 启动器）
5. **Advanced Options**（高级选项）：
   - ✅ Install for all users（为所有用户安装）
   - ✅ Associate files with Python（关联 .py 文件）
   - ✅ Create shortcuts for installed applications
   - ✅ Add Python to environment variables（添加到环境变量）
   - ✅ Precompile standard library
   - 安装路径：默认 `C:\Program Files\Python311\` 或自定义
6. 点击 "Install" 开始安装
7. 安装完成后，点击 "Disable path length limit"（禁用路径长度限制）

---

#### macOS 系统安装

**方法1：使用 Homebrew（推荐）**

```bash
# 安装 Homebrew（如果未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Python
brew install python@3.11

# 验证安装
python3 --version
```

**方法2：官网下载**
- 访问 https://www.python.org/downloads/macos/
- 下载 `.pkg` 安装包
- 双击安装

---

#### Linux 系统安装

**Ubuntu/Debian：**

```bash
# 更新包列表
sudo apt update

# 安装 Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip

# 验证安装
python3.11 --version
```

**Fedora/RHEL：**

```bash
sudo dnf install python3.11
```

---

### 4. 环境变量配置

**Windows 环境变量验证：**

如果安装时勾选了 "Add Python to PATH"，环境变量会自动配置。

**手动配置环境变量（如果未自动配置）：**

1. 右键"此电脑" → "属性" → "高级系统设置" → "环境变量"
2. 在"系统变量"中找到 `Path`，点击"编辑"
3. 添加以下两个路径：
   ```
   C:\Program Files\Python311\
   C:\Program Files\Python311\Scripts\
   ```
4. 点击"确定"保存
5. **重启终端**或重启电脑使环境变量生效

---

### 5. 验证安装

打开终端（Windows: PowerShell / CMD，macOS/Linux: Terminal），执行以下命令验证：

```bash
# 查看 Python 版本
python --version
# 输出示例：Python 3.11.7

# 或
python3 --version

# 查看 pip 版本
pip --version
# 输出示例：pip 23.3.1 from C:\Program Files\Python311\Lib\site-packages\pip (python 3.11)
```

**如果提示"python 不是内部或外部命令"：**
- 检查环境变量是否配置正确
- 重启终端或电脑
- 使用 `python3` 命令代替 `python`

---

## 💻 代码实战

### 验证 Python 安装

创建第一个 Python 程序，验证安装是否成功。

**步骤1：创建 Python 文件**

创建文件 `hello_python.py`：

```python
# hello_python.py
# Python 安装验证脚本

import sys
import platform

print("=" * 50)
print("🎉 Python 安装成功！")
print("=" * 50)

# 显示 Python 版本信息
print(f"\n📌 Python 版本：{sys.version}")
print(f"📌 Python 版本号：{platform.python_version()}")
print(f"📌 Python 解释器路径：{sys.executable}")

# 显示系统信息
print(f"\n💻 操作系统：{platform.system()}")
print(f"💻 系统版本：{platform.release()}")
print(f"💻 处理器架构：{platform.machine()}")

# 显示 Python 搜索路径
print(f"\n📂 Python 模块搜索路径：")
for path in sys.path:
    if path:  # 过滤空路径
        print(f"   - {path}")

print("\n" + "=" * 50)
print("✅ Python 环境检查完成！")
print("=" * 50)
```

**步骤2：运行脚本**

```bash
# 运行脚本
python hello_python.py

# 或
python3 hello_python.py
```

**预期输出：**
```
==================================================
🎉 Python 安装成功！
==================================================

📌 Python 版本：3.11.7 (tags/v3.11.7:fa7a6f2, Dec  4 2023, 19:24:49) [MSC v.1937 64 bit (AMD64)]
📌 Python 版本号：3.11.7
📌 Python 解释器路径：C:\Program Files\Python311\python.exe

💻 操作系统：Windows
💻 系统版本：10
💻 处理器架构：AMD64

📂 Python 模块搜索路径：
   - E:\学习用\视觉教学
   - C:\Program Files\Python311\python311.zip
   - C:\Program Files\Python311\Lib
   - C:\Program Files\Python311
   - C:\Program Files\Python311\Lib\site-packages

==================================================
✅ Python 环境检查完成！
==================================================
```

---

### pip 包管理器使用

**pip** 是 Python 的包管理工具，用于安装和管理第三方库。

#### 基本命令

```bash
# 查看 pip 版本
pip --version

# 升级 pip 到最新版本
python -m pip install --upgrade pip

# 安装包
pip install package_name

# 安装指定版本的包
pip install package_name==1.2.3

# 升级包
pip install --upgrade package_name

# 卸载包
pip uninstall package_name

# 查看已安装的包
pip list

# 查看包的详细信息
pip show package_name

# 导出已安装的包列表
pip freeze > requirements.txt

# 从文件安装包
pip install -r requirements.txt
```

---

#### 配置国内镜像源（加速下载）

**临时使用镜像源：**

```bash
# 使用清华镜像源安装包
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**永久配置镜像源：**

**Windows 配置：**

创建文件 `C:\Users\你的用户名\pip\pip.ini`：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
```

**macOS/Linux 配置：**

创建文件 `~/.pip/pip.conf`：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
```

**常用国内镜像源：**

| 镜像源 | 地址 |
|--------|------|
| **清华大学** | https://pypi.tuna.tsinghua.edu.cn/simple |
| **阿里云** | https://mirrors.aliyun.com/pypi/simple/ |
| **中科大** | https://pypi.mirrors.ustc.edu.cn/simple/ |
| **豆瓣** | https://pypi.douban.com/simple/ |

---

#### pip 使用示例

创建文件 `test_pip.py` 测试安装包：

```python
# test_pip.py
# 测试 pip 安装的包

try:
    import sys
    print(f"Python 版本: {sys.version}\n")

    # 测试导入常用内置模块
    import os
    import math
    import json
    import datetime

    print("✅ 内置模块测试通过：")
    print("   - os (操作系统接口)")
    print("   - math (数学函数)")
    print("   - json (JSON 处理)")
    print("   - datetime (日期时间)")

    # 测试 pip 安装的包（示例）
    print("\n📦 尝试导入第三方包...")

    # 注意：这些包需要先通过 pip install 安装
    # pip install requests
    try:
        import requests
        print(f"   ✅ requests {requests.__version__}")
    except ImportError:
        print("   ❌ requests 未安装（可选）")

    print("\n🎉 Python 包管理系统正常！")

except Exception as e:
    print(f"❌ 错误：{e}")
```

**运行脚本：**

```bash
python test_pip.py
```

---

#### 实战：安装第一个第三方包

```bash
# 安装 requests 库（HTTP 请求库）
pip install requests

# 验证安装
python -c "import requests; print(requests.__version__)"
```

创建 `test_requests.py` 测试：

```python
# test_requests.py
import requests

# 发送 HTTP GET 请求
response = requests.get("https://www.python.org")

print(f"状态码：{response.status_code}")
print(f"响应头：{response.headers['Content-Type']}")
print(f"网页内容长度：{len(response.text)} 字符")

if response.status_code == 200:
    print("\n✅ requests 库工作正常！")
```

---

## 🎯 总结

本节学习了 Python 的完整安装配置流程：

✅ **选择 Python 版本**：推荐 Python 3.11.x
✅ **下载与安装**：官网下载，勾选"Add to PATH"
✅ **验证安装**：`python --version` 和 `pip --version`
✅ **配置 pip**：设置国内镜像源加速下载
✅ **掌握 pip 命令**：安装、升级、卸载第三方包

**下一步：**
👉 [03 - 虚拟环境管理](../03-虚拟环境管理/README.md)

---

## 📚 参考资料

- [Python 官网](https://www.python.org/)
- [Python 文档（中文）](https://docs.python.org/zh-cn/3/)
- [pip 文档](https://pip.pypa.io/en/stable/)
- [清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
