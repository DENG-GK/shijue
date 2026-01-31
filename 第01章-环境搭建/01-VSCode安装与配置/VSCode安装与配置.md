# VSCode 安装与配置

> Visual Studio Code 是微软推出的免费、开源、跨平台的现代化代码编辑器，是 Python 开发的首选工具之一。

---

## 📖 理论部分

### 1. VSCode 简介

**Visual Studio Code (VSCode)** 是一款轻量级但功能强大的源代码编辑器，支持：

- ✅ **跨平台**：Windows、macOS、Linux
- ✅ **丰富的扩展**：海量插件支持各种编程语言
- ✅ **智能代码补全**：IntelliSense 智能提示
- ✅ **内置 Git**：版本控制集成
- ✅ **调试功能**：强大的 Debug 工具
- ✅ **终端集成**：内置终端，无需切换窗口

**为什么选择 VSCode 进行 Python 开发？**
- 免费开源
- 启动速度快
- 扩展丰富（Python、Jupyter、Pylance 等）
- 调试体验优秀
- 社区活跃，文档完善

---

### 2. 下载与安装

#### Windows 系统安装

**步骤1：下载**
- 访问官网：https://code.visualstudio.com/
- 点击 "Download for Windows" 下载安装包（约 80MB）
- 或直接下载：https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-user

**步骤2：安装**
1. 双击下载的 `VSCodeUserSetup-x64-x.xx.x.exe`
2. 选择安装路径（建议默认）
3. **重要选项**（务必勾选）：
   - ✅ 添加到 PATH（重启后生效）
   - ✅ 创建桌面快捷方式
   - ✅ 通过 Code 打开（右键菜单集成）
   - ✅ 将 Code 注册为受支持文件类型的编辑器
4. 点击"安装"，等待完成
5. 勾选"启动 Visual Studio Code"，点击"完成"

**步骤3：首次启动**
- 选择主题：Dark+（推荐）、Light+
- 选择语言：简体中文（首次需要安装中文语言包）

---

#### macOS 系统安装

```bash
# 使用 Homebrew 安装（推荐）
brew install --cask visual-studio-code

# 或手动下载 .dmg 文件安装
# 访问 https://code.visualstudio.com/
```

---

#### Linux 系统安装

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install code

# Fedora/RHEL
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
sudo dnf install code
```

---

### 3. 界面介绍

VSCode 主界面分为五个区域：

```
┌─────────────────────────────────────────────┐
│  菜单栏 (File, Edit, View...)              │
├──────┬──────────────────────────────────────┤
│      │                                      │
│ 活动 │         编辑器区域                    │
│ 栏   │      (打开的文件标签页)               │
│      │                                      │
│ (侧  ├──────────────────────────────────────┤
│  边  │                                      │
│  栏) │         面板区域                     │
│      │   (终端、输出、调试控制台)            │
└──────┴──────────────────────────────────────┘
│              状态栏                         │
└─────────────────────────────────────────────┘
```

**关键区域说明：**

1. **活动栏**（最左侧）
   - 📁 资源管理器（查看项目文件）
   - 🔍 搜索（全局搜索）
   - 🔀 源代码管理（Git）
   - 🐛 运行和调试
   - 🧩 扩展

2. **侧边栏**
   - 显示活动栏选中的内容
   - 文件树、搜索结果、Git 变更等

3. **编辑器区域**
   - 代码编辑的主要区域
   - 支持分屏、标签页

4. **面板区域**
   - 终端（Terminal）
   - 输出（Output）
   - 问题（Problems）
   - 调试控制台（Debug Console）

5. **状态栏**
   - 显示当前文件信息、编码、行列号
   - Python 解释器选择
   - Git 分支信息

---

### 4. Python 扩展安装

VSCode 需要安装 Python 扩展才能支持 Python 开发。

**必装扩展：**

#### 1. Python (Microsoft 官方)
- **功能**：Python 语言支持、智能提示、调试
- **安装方法**：
  1. 点击左侧活动栏的"扩展"图标 (Ctrl+Shift+X)
  2. 搜索 "Python"
  3. 找到 Microsoft 发布的 "Python"
  4. 点击 "Install" 安装

#### 2. Pylance (微软官方)
- **功能**：高性能 Python 语言服务器，提供更好的类型检查和代码补全
- **安装方法**：同上，搜索 "Pylance"

**推荐扩展：**

| 扩展名称 | 功能 | 说明 |
|---------|------|------|
| **Jupyter** | Jupyter Notebook 支持 | 在 VSCode 中运行 .ipynb 文件 |
| **Python Indent** | 自动缩进 | 智能 Python 缩进 |
| **autoDocstring** | 自动生成文档字符串 | 快速生成函数注释 |
| **Better Comments** | 注释高亮 | 让注释更美观 |
| **Error Lens** | 行内错误提示 | 在代码行直接显示错误 |
| **Git Graph** | Git 可视化 | 可视化 Git 提交历史 |
| **Chinese (Simplified)** | 简体中文语言包 | 界面汉化 |

**快速安装推荐扩展：**
```bash
# 打开 VSCode 终端 (Ctrl + `)，执行以下命令
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension njpwerner.autodocstring
code --install-extension aaron-bond.better-comments
```

---

### 5. 推荐配置

VSCode 的配置文件是 JSON 格式，可以自定义各种行为。

**打开设置：**
- 方法1：`文件` → `首选项` → `设置`
- 方法2：快捷键 `Ctrl + ,`
- 方法3：命令面板 `Ctrl + Shift + P` → 输入 "settings"

**设置类型：**
- **用户设置**：全局生效
- **工作区设置**：仅当前项目生效

---

## 💻 代码实战

### VSCode 推荐配置文件

创建 **settings.json** 配置文件（用户设置）：

**打开方式：**
1. 按 `Ctrl + Shift + P` 打开命令面板
2. 输入 "Preferences: Open Settings (JSON)"
3. 回车打开 settings.json

**推荐配置内容：**

```json
{
  // ========== 编辑器基础设置 ==========
  "editor.fontSize": 14,                    // 字体大小
  "editor.fontFamily": "Consolas, 'Courier New', monospace",
  "editor.tabSize": 4,                      // Tab 缩进为 4 空格
  "editor.insertSpaces": true,              // 使用空格而非 Tab
  "editor.wordWrap": "on",                  // 自动换行
  "editor.formatOnSave": true,              // 保存时自动格式化
  "editor.minimap.enabled": true,           // 显示代码缩略图
  "editor.rulers": [80, 120],               // 显示80和120字符标尺
  "editor.renderWhitespace": "boundary",    // 显示空白字符

  // ========== 文件设置 ==========
  "files.autoSave": "afterDelay",           // 自动保存
  "files.autoSaveDelay": 1000,              // 延迟1秒保存
  "files.encoding": "utf8",                 // 文件编码
  "files.eol": "\n",                        // 行尾符号（LF）
  "files.trimTrailingWhitespace": true,     // 保存时删除行尾空格

  // ========== Python 专项设置 ==========
  "python.defaultInterpreterPath": "python",
  "python.linting.enabled": true,           // 启用代码检查
  "python.linting.pylintEnabled": false,    // 禁用 pylint（较慢）
  "python.linting.flake8Enabled": true,     // 启用 flake8
  "python.formatting.provider": "black",    // 使用 black 格式化
  "python.analysis.typeCheckingMode": "basic", // 类型检查

  // ========== 终端设置 ==========
  "terminal.integrated.fontSize": 13,
  "terminal.integrated.defaultProfile.windows": "PowerShell",

  // ========== 外观设置 ==========
  "workbench.colorTheme": "Default Dark+",  // 主题
  "workbench.iconTheme": "vs-seti",         // 文件图标主题

  // ========== Jupyter 设置 ==========
  "jupyter.askForKernelRestart": false,
  "jupyter.interactiveWindow.textEditor.executeSelection": true
}
```

**配置文件保存位置：**
- Windows: `C:\Users\你的用户名\AppData\Roaming\Code\User\settings.json`
- macOS: `~/Library/Application Support/Code/User/settings.json`
- Linux: `~/.config/Code/User/settings.json`

---

### Python 开发环境配置

**1. 选择 Python 解释器**

创建一个 Python 文件测试：

```python
# test_hello.py
print("Hello, VSCode!")
print("Python 版本测试")

import sys
print(f"Python 解释器路径: {sys.executable}")
print(f"Python 版本: {sys.version}")
```

**选择解释器步骤：**
1. 打开上面的 Python 文件
2. 按 `Ctrl + Shift + P` 打开命令面板
3. 输入 "Python: Select Interpreter"
4. 选择已安装的 Python 版本（如 Python 3.11.x）
5. 状态栏左下角会显示当前选择的 Python 版本

---

**2. 运行 Python 代码**

VSCode 提供多种运行方式：

| 方式 | 快捷键 | 说明 |
|------|--------|------|
| **右键菜单** | - | 右键代码 → "在终端中运行 Python 文件" |
| **运行按钮** | - | 右上角播放按钮 ▶️ |
| **快捷键** | `Ctrl + F5` | 直接运行（不调试） |
| **调试运行** | `F5` | 调试模式运行 |
| **选中代码运行** | `Shift + Enter` | 在交互窗口运行选中代码 |

---

**3. 调试配置**

创建 `.vscode/launch.json` 调试配置：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: 当前文件",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}
```

**使用调试功能：**
1. 在代码行号左侧点击，设置断点（红点）
2. 按 `F5` 开始调试
3. 使用调试工具栏控制执行：
   - **继续** (F5)：运行到下一个断点
   - **单步跳过** (F10)：执行当前行，不进入函数
   - **单步调试** (F11)：进入函数内部
   - **单步跳出** (Shift+F11)：跳出当前函数
   - **停止** (Shift+F5)：终止调试

---

**4. 常用快捷键**

| 功能 | Windows/Linux | macOS |
|------|---------------|-------|
| **命令面板** | `Ctrl + Shift + P` | `Cmd + Shift + P` |
| **快速打开文件** | `Ctrl + P` | `Cmd + P` |
| **切换终端** | `Ctrl + `` | `Ctrl + `` |
| **格式化文档** | `Shift + Alt + F` | `Shift + Option + F` |
| **跳转到定义** | `F12` | `F12` |
| **查找所有引用** | `Shift + F12` | `Shift + F12` |
| **重命名符号** | `F2` | `F2` |
| **多光标编辑** | `Alt + Click` | `Option + Click` |
| **复制当前行** | `Shift + Alt + ↓/↑` | `Shift + Option + ↓/↑` |
| **注释/取消注释** | `Ctrl + /` | `Cmd + /` |

---

## 🎯 总结

本节学习了 VSCode 的完整配置流程：

✅ **安装 VSCode**：跨平台免费编辑器
✅ **安装 Python 扩展**：Python、Pylance、Jupyter
✅ **配置 settings.json**：自定义编辑器行为
✅ **配置 Python 环境**：选择解释器、调试设置
✅ **掌握快捷键**：提高开发效率

**下一步：**
👉 [02 - Python安装与配置](../02-Python安装与配置/README.md)

---

## 📚 参考资料

- [VSCode 官网](https://code.visualstudio.com/)
- [VSCode Python 教程](https://code.visualstudio.com/docs/python/python-tutorial)
- [VSCode 快捷键速查](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf)
- [Python 扩展文档](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
