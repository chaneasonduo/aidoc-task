# DashScope API 客户端使用指南

## 概述

本项目提供了一个封装好的DashScope API客户端，可以方便地调用阿里云DashScope的文本生成服务。

## 主要组件

### 1. DashScopeAPIClient

独立的API客户端类，封装了与DashScope API的HTTP通信逻辑。

#### 主要功能：
- 自动处理API密钥管理
- 构建标准化的请求payload
- 处理API响应和错误
- 支持自定义参数配置

#### 使用示例：

```python
from new_model import DashScopeAPIClient
from langchain_core.messages import SystemMessage, HumanMessage

# 创建客户端
client = DashScopeAPIClient(
    api_key="your_api_key",  # 可选，默认从环境变量获取
    api_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
    timeout=60
)

# 准备消息
messages = [
    SystemMessage(content="你是一个专业的AI助手"),
    HumanMessage(content="你好，请介绍一下你自己")
]

# 调用API
response = client.call_api(
    messages=messages,
    model_name="qwen-turbo",
    temperature=0.7,
    max_tokens=100
)

print(response)
```

### 2. CustomChatModel

基于LangChain的聊天模型实现，集成了DashScope API客户端。

#### 主要特性：
- 完全兼容LangChain生态
- 支持标准的LangChain接口（invoke, stream等）
- 自动处理消息格式转换
- 支持流式输出（待实现）

#### 使用示例：

```python
from new_model import CustomChatModel
from langchain_core.messages import SystemMessage, HumanMessage

# 创建模型实例
model = CustomChatModel(
    model_name="qwen-turbo",
    temperature=0.7,
    max_tokens=2000,
    api_key="your_api_key"  # 可选，默认从环境变量获取
)

# 方式1：使用_generate方法
messages = [
    SystemMessage(content="你是一个专业的AI助手"),
    HumanMessage(content="你好，请介绍一下你自己")
]
result = model._generate(messages)
print(result.generations[0].message.content)

# 方式2：使用invoke方法（推荐）
response = model.invoke("你好，请介绍一下你自己")
print(response.content)
```

## 环境配置

### 1. 设置API密钥

```bash
# 方法1：设置环境变量
export DASHSCOPE_API_KEY="your_api_key_here"

# 方法2：在代码中直接提供
model = CustomChatModel(api_key="your_api_key_here")
```

### 2. 安装依赖

```bash
pip install langchain-core requests
```

## API参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | "qwen-turbo" | 模型名称 |
| `temperature` | float | 0.7 | 温度参数，控制输出的随机性 |
| `max_tokens` | int | 2000 | 最大生成token数 |
| `api_key` | str | None | API密钥 |
| `api_url` | str | DashScope默认URL | API端点 |
| `timeout` | int | 60 | 请求超时时间（秒） |

### 支持的模型

- `qwen-turbo`: 通义千问Turbo版本
- `qwen-plus`: 通义千问Plus版本
- `qwen-max`: 通义千问Max版本

## 错误处理

客户端会自动处理以下错误：

1. **API密钥错误**: 如果未提供有效的API密钥
2. **网络错误**: 请求超时或网络连接问题
3. **API错误**: 服务器返回非200状态码
4. **响应解析错误**: 响应格式不符合预期

## 测试

运行测试脚本验证功能：

```bash
python appserver/models/test_dashscope_api.py
```

## 扩展功能

### 1. 添加流式输出支持

```python
# 在CustomChatModel中实现_stream方法
def _stream(self, messages, stop=None, run_manager=None, **kwargs):
    # 实现流式输出逻辑
    pass
```

### 2. 添加异步支持

```python
# 在CustomChatModel中实现_agenerate和_astream方法
async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
    # 实现异步生成逻辑
    pass
```

### 3. 添加更多模型支持

在`DashScopeAPIClient`中添加对不同模型API格式的支持。

## 注意事项

1. **API密钥安全**: 不要在代码中硬编码API密钥，使用环境变量
2. **请求频率**: 注意API的调用频率限制
3. **错误重试**: 对于网络错误，可以考虑添加重试机制
4. **日志记录**: 在生产环境中添加适当的日志记录

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！ 