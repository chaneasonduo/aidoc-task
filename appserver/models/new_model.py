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

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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
        # 如果没有提供api_key，从环境变量获取
        if self.api_key is None:
            import os
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
            if self.api_key is None:
                raise ValueError("请设置DASHSCOPE_API_KEY环境变量或直接提供api_key参数")
    
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
        # 构建API请求payload
        payload = {
            "model": self.model_name,
            "input": {
                "messages": [convert_message_to_dict(msg) for msg in messages]
            },
            "parameters": {
                "result_format": "message",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        }
        
        # 添加额外参数
        if kwargs:
            payload["parameters"].update(kwargs)
        
        # 发送API请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            raise ValueError(f"API请求失败: {response.status_code} - {response.text}")
        
        # 解析响应
        response_data = response.json()
        content = response_data["output"]["choices"][0]["message"]["content"]
        
        # 创建ChatResult
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
    
    # def _stream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[Any] = None,
    #     **kwargs: Any,
    # ) -> Iterator[ChatGenerationChunk]:
    #     """
    #     流式生成聊天回复
        
    #     Args:
    #         messages: 输入消息列表
    #         stop: 停止词列表
    #         run_manager: 运行管理器
    #         **kwargs: 其他参数
            
    #     Yields:
    #         ChatGenerationChunk: 聊天生成块
    #     """
    #     pass
    
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
    print("====== _generate ======")
    result = model._generate([SystemMessage(content="你是一个专业的AI助手"), HumanMessage(content="你好，请用简洁的语言介绍一下你自己")])   
    print(result)
    print("====== invoke ======")
    result = model.invoke("你好，请用简洁的语言介绍一下你自己")
    print(result)