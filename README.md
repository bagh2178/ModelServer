# 🤖 ModelServer

[![English](https://img.shields.io/badge/README-English-blue)](README_EN.md)

> **ModelServer** 是一个基于 web 服务，用于在上位机部署模型或多模型之间的环境隔离，并运行客户端通过 API 调用服务的框架。

## 📦 安装

### 方法一：通过 pip 安装
```bash
pip install git+https://github.com/bagh2178/ModelServer.git
```

### 方法二：克隆仓库安装
```bash
git clone https://github.com/bagh2178/ModelServer.git
pip install -e ModelServer/
```

## 🚀 运行

### 🖥️ 服务器端

#### 方式一：快速启动
```bash
python scripts/quickstart_server/hexmove.py
```

#### 方式二：编程方式启动
```python
from ModelServer import start_server

# 创建模型实例
your_model = YourModelClass(param1="value1", param2="value2")  # 根据需要传入参数

# 启动服务器，传入模型实例
start_server(your_model, port=7002)  # 传入已初始化的模型实例和服务器端口
```

### 💻 客户端

#### 初始化客户端

**方式一：快速启动**
```python
from ModelServer import hexmove
```

**方式二：编程方式启动**
```python
from ModelServer import ModelClient

# 本地模型
your_model = ModelClient('http://localhost:7002')

# 远程模型
your_model = ModelClient('http://166.111.73.73:7002')
```

#### 同步模式用法示例

同一个客户端对象（如 `your_model` 或 `hexmove`）既支持同步模式，也支持异步模式。以下为同步模式的调用方式，直接调用方法即可获得结果。

你可以像操作普通 Python 类实例一样，直接通过 `your_model` 在客户端调用 `YourModelClass` 中定义的方法。

```python
# 📷 获取 RGB-D 图像
rgb_image, depth_image, pose, timestamp = hexmove.get_rgbd_image('FemtoBolt_down', format='JPEG', pose=True)
```

#### 异步模式用法示例

异步模式下，调用 `your_model.async_mode.xxx()` 会立即返回一个 `proxy`。调用它即可获取最终结果：

```python
# 通过 async_mode 异步调用
proxy = hexmove.async_mode.get_rgbd_image('FemtoBolt_down', format='JPEG', pose=True)

# 此处可以执行其他代码

# 等待并获取结果（阻塞等待结果）
rgb_image, depth_image, pose, timestamp = proxy()  # 或 proxy.result
```

## 🔧 特性

- 🌐 **Web 服务架构** - 基于 HTTP API 的模型服务
- 🔒 **环境隔离** - 支持多模型环境隔离部署
- 📡 **远程调用** - 支持本地和远程模型调用
- ⏩ **异步支持** - 支持异步模型推理与调用
- 🔌 **易于集成** - 简单的客户端 API 接口

## 📝 使用示例

### 基本使用流程

1. **启动服务器** 🚀
   ```bash
   python scripts/quickstart_server/hexmove.py
   ```

2. **连接客户端** 🔗
   ```python
   from ModelServer import hexmove
   ```

3. **调用模型方法** 📞
   ```python
   # 获取传感器数据
   rgb, depth, pose, ts = hexmove.get_rgbd_image('camera_id')
   ```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。
