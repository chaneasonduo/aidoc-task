# 1. 导入必要的依赖
#    - langchain_core.messages 相关的消息类型
#    - langchain_core.language_models.base.BaseChatModel
#    - langchain_core.outputs 相关类型
#    - typing/Pydantic等

# 2. 定义自定义ChatModel类，继承自BaseChatModel

# 3. 实现必要的属性和方法
#    - _llm_type: 返回模型类型字符串
#    - _generate: 实现核心的消息生成逻辑（输入为消息列表，输出为ChatResult）
#    - _stream/astream: （可选）实现流式输出
#    - _identifying_params: 返回模型的唯一参数（如model_name等）

# 4. 消息格式转换
#    - 实现消息对象与API格式之间的转换函数（如 message_to_dict, dict_to_message）

# 5. API请求逻辑
#    - 实现与实际LLM服务（如DashScope、OpenAI等）的HTTP请求逻辑
#    - 支持同步和异步请求

# 6. 支持参数配置
#    - 支持如model_name、temperature、max_tokens等常用参数
#    - 支持API Key等安全配置

# 7. 错误处理
#    - 对API异常、参数异常等进行友好处理

# 8. 文档字符串和类型注解
#    - 为每个方法和类添加详细的docstring和类型注解，便于IDE和文档生成

# 9. 使用示例
#    - 在文件末尾或单独文件，给出如何实例化和调用自定义ChatModel的示例

# 10. （可选）测试用例
#    - 编写简单的单元测试，验证核心功能

# 备注：
# - 全程遵循LangChain官方接口和类型要求，确保与LangChain生态兼容
# - 代码风格清晰，注释详细，便于后续维护和扩展

import json
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def convert_message_to_dict(message: BaseMessage) -> Dict[str, str]:
    """将LangChain消息对象转换为DashScope API格式"""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": str(message.content)}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": str(message.content)}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": str(message.content)}
    else:
        return {"role": "user", "content": str(message.content)}


class DashScopeAPIClient:
    """
    DashScope API客户端，封装HTTP请求逻辑
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        timeout: int = 60
    ):
        """
        初始化DashScope API客户端
        
        Args:
            api_key: API密钥，如果为None则从环境变量获取
            api_url: API端点URL
            timeout: 请求超时时间（秒）
        """
        self.api_url = api_url
        self.timeout = timeout
        
        # 获取API密钥
        if api_key is None:
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
            if self.api_key is None:
                raise ValueError("请设置DASHSCOPE_API_KEY环境变量或直接提供api_key参数")
        else:
            self.api_key = api_key
    
    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _build_payload(
        self,
        messages: List[BaseMessage],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2000,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        构建API请求payload
        
        Args:
            messages: 消息列表
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            Dict: API请求payload
        """
        payload = {
            "model": model_name,
            "input": {
                "messages": [convert_message_to_dict(msg) for msg in messages]
            },
            "parameters": {
                "result_format": "message",
                "temperature": temperature,
                "max_tokens": max_tokens,
                # 增量输出
                "incremental_output": "true"
            }
        }
        
        # 添加额外参数
        if kwargs:
            payload["parameters"].update(kwargs)
        
        return payload
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """
        解析API响应
        
        Args:
            response_data: API响应数据
            
        Returns:
            str: 解析出的内容
        """
        return response_data["output"]["choices"][0]["message"]["content"]
    
    def call_api(
        self,
        messages: List[BaseMessage],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2000,
        **kwargs: Any
    ) -> str:
        """
        调用DashScope API
        
        Args:
            messages: 消息列表
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            str: API返回的内容
            
        Raises:
            ValueError: API请求失败时抛出
        """
        # 构建请求数据
        payload = self._build_payload(
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        headers = self._build_headers()
        
        # 发送API请求
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise ValueError(f"API请求失败: {response.status_code} - {response.text}")
        
        # 解析响应
        response_data = response.json()
        return self._parse_response(response_data)
    
    def call_api_stream(
        self,
        messages: List[BaseMessage],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2000,
        **kwargs: Any
    ) -> Iterator[str]:
        """
        流式调用DashScope API
        
        Args:
            messages: 消息列表
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Yields:
            str: 流式返回的内容片段
            
        Raises:
            ValueError: API请求失败时抛出
        """
        # 构建请求数据
        payload = self._build_payload(
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        headers = self._build_headers()
        headers["Accept"] = "text/event-stream"
        headers["X-DashScope-SSE"] = "enable"

        # 发送流式API请求
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise ValueError(f"API请求失败: {response.status_code} - {response.text}")
        
        # 处理Server-Sent Events (SSE)格式的流式响应
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data:'):
                    data = line[5:]  # 移除 'data: ' 前缀
                    if data.strip() == '[DONE]':
                        break
                    try:
                        chunk_data = json.loads(data)
                        # 解析官方API返回格式
                        if 'output' in chunk_data and 'choices' in chunk_data['output']:
                            choice = chunk_data['output']['choices'][0]
                            if 'message' in choice and 'content' in choice['message']:
                                content = choice['message']['content']
                                if content:
                                    yield content
                    except json.JSONDecodeError:
                        continue


class CustomChatModel(BaseChatModel):
    """
    自定义ChatModel，继承自BaseChatModel
    """
    
    # 模型配置参数
    model_name: str = "qwen-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = 2000
    
    # API配置
    api_key: Optional[str] = None
    api_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化API客户端
        self._api_client = DashScopeAPIClient(
            api_key=self.api_key,
            api_url=self.api_url
        )
    
    @property
    def _llm_type(self) -> str:
        """返回模型类型标识"""
        return "custom_chat_model"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于标识模型的参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        生成聊天回复的核心方法
        
        Args:
            messages: 输入消息列表
            stop: 停止词列表
            run_manager: 运行管理器
            **kwargs: 其他参数
            
        Returns:
            ChatResult: 聊天结果
        """
        # 使用API客户端调用API
        content = self._api_client.call_api(
            messages=messages,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        
        # 创建ChatResult
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        流式生成聊天回复
        
        Args:
            messages: 输入消息列表
            stop: 停止词列表
            run_manager: 运行管理器
            **kwargs: 其他参数
            
        Yields:
            ChatGenerationChunk: 聊天生成块
        """
        # 使用API客户端进行流式调用
        result = self._api_client.call_api_stream(
            messages=messages,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )

        for chunk in result:
            # 创建ChatGenerationChunk，直接使用流式返回的内容
            generation_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=chunk)
            )
            yield generation_chunk
    
    # async def _agenerate(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[Any] = None,
    #     **kwargs: Any,
    # ) -> ChatResult:
    #     """
    #     异步生成聊天回复
        
    #     Args:
    #         messages: 输入消息列表
    #         stop: 停止词列表
    #         run_manager: 运行管理器
    #         **kwargs: 其他参数
            
    #     Returns:
    #         ChatResult: 聊天结果
    #     """
    #     pass
    
    # async def _astream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[Any] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[ChatGenerationChunk]:
    #     """
    #     异步流式生成聊天回复
        
    #     Args:
    #         messages: 输入消息列表
    #         stop: 停止词列表
    #         run_manager: 运行管理器
    #         **kwargs: 其他参数
            
    #     Yields:
    #         ChatGenerationChunk: 聊天生成块
    #     """
    #     pass


if __name__ == "__main__":
    model = CustomChatModel()
    # print("====== _generate ======")
    # result = model._generate([SystemMessage(content="你是一个专业的AI助手"), HumanMessage(content="你好，请用简洁的语言介绍一下你自己")])   
    # print(result)
    # print("====== invoke ======")
    # result = model.invoke("你好，请用简洁的语言介绍一下你自己")
    # print(result)
    result = model._stream([SystemMessage(content="你是一个专业的AI助手"), HumanMessage(content="你好，请用简洁的语言介绍一下你自己")])   
    for chunk in result:
        print(chunk)
