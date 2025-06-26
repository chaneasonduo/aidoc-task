from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Type, Union, Literal

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env

import dashscope
from dashscope import Generation


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """
    将LangChain消息对象转换为DashScope API所需的字典格式
    
    Args:
        message: LangChain消息对象
        
    Returns:
        dict: 包含role和content的字典
    """
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    else:
        raise ValueError(f"不支持的消息类型: {message}")


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """
    将字典转换回LangChain消息对象
    
    Args:
        _dict: 包含role和content的字典
        
    Returns:
        BaseMessage: LangChain消息对象
    """
    role = _dict.get("role")
    content = _dict.get("content", "")
    
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        return AIMessage(content=content)


class CustomDashScopeLLM(LLM):
    """
    自定义的DashScope LLM包装器，支持文本补全和聊天两种模式
    
    特性：
    - 支持流式和非流式输出
    - 支持同步和异步调用
    - 兼容LangChain生态
    """
    
    # 基础配置
    model_name: str = "qwen-turbo"
    "模型名称，如 qwen-turbo, qwen-max 等"
    
    mode: Literal["completion", "chat"] = "chat"
    "运行模式：completion(补全) 或 chat(聊天)"
    
    # 生成参数
    temperature: float = 0.7
    "采样温度，0-2之间，值越大随机性越强"
    
    max_tokens: Optional[int] = 2000
    "最大生成长度"
    
    top_p: float = 0.8
    "核采样参数，0-1之间，控制生成多样性"
    
    # 配置
    dashscope_api_key: Optional[str] = None
    "DashScope API密钥，如果不提供会从环境变量DASHSCOPE_API_KEY读取"
    
    streaming: bool = False
    "是否启用流式输出"
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """验证环境配置"""
        values["dashscope_api_key"] = get_from_dict_or_env(
            values, "dashscope_api_key", "DASHSCOPE_API_KEY"
        )
        dashscope.api_key = values["dashscope_api_key"]
        return values
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "dashscope"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        同步调用方法 - 处理文本补全
        
        Args:
            prompt: 输入提示
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        if self.mode == "chat":
            messages = [HumanMessage(content=prompt)]
            result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            return result.generations[0].message.content
        else:
            if self.streaming:
                return "".join(
                    chunk.text for chunk in 
                    self._stream(prompt, stop=stop, run_manager=run_manager, **kwargs)
                )
                
            response = self._call_dashscope(
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.output.choices[0].message.content
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        生成聊天回复
        
        Args:
            messages: 消息列表
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Returns:
            ChatResult: 聊天结果
        """
        if self.streaming:
            return self._generate_stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            
        dashscope_messages = [_convert_message_to_dict(msg) for msg in messages]
        response = self._call_dashscope(
            messages=dashscope_messages,
            **kwargs
        )
        
        message = _convert_dict_to_message({
            "role": "assistant",
            "content": response.output.choices[0].message.content,
        })
        
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        流式文本生成
        
        Args:
            prompt: 输入提示
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Yields:
            ChatGenerationChunk: 生成的内容块
        """
        if self.mode == "chat":
            messages = [HumanMessage(content=prompt)]
            for chunk in self._generate_stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=chunk.message.content)
                )
            return
            
        response = Generation.call(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs,
        )
        
        for chunk in response:
            if chunk.status_code != 200:
                raise ValueError(f"DashScope API错误: {chunk}")
                
            if hasattr(chunk, 'output') and hasattr(chunk.output, 'choices'):
                content = chunk.output.choices[0].message.content
                if content:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=content)
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(content, chunk=chunk)
                    yield chunk
    
    def _generate_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        流式生成聊天回复
        
        Args:
            messages: 消息列表
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他参数
            
        Returns:
            ChatResult: 聊天结果
        """
        dashscope_messages = [_convert_message_to_dict(msg) for msg in messages]
        
        response = Generation.call(
            model=self.model_name,
            messages=dashscope_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs,
        )
        
        full_response = ""
        for chunk in response:
            if chunk.status_code != 200:
                raise ValueError(f"DashScope API错误: {chunk}")
                
            if hasattr(chunk, 'output') and hasattr(chunk.output, 'choices'):
                content = chunk.output.choices[0].message.content
                if content:
                    full_response += content
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=content)
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(content, chunk=chunk)
        
        message = AIMessage(content=full_response)
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _call_dashscope(self, messages: List[dict], **kwargs: Any) -> Any:
        """
        调用DashScope API的通用方法
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            Any: API响应
        """
        response = Generation.call(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        
        if response.status_code != 200:
            raise ValueError(f"DashScope API错误: {response}")
            
        return response


# 使用示例
if __name__ == "__main__":
    # 1. 文本补全模式
    llm = CustomDashScopeLLM(
        mode="completion",
        model_name="qwen-turbo",
        temperature=0.7,
        streaming=True
    )
    
    # 2. 聊天模式
    chat = CustomDashScopeLLM(
        mode="chat",
        model_name="qwen-turbo",
        temperature=0.7
    )
    
    # 3. 使用示例
    # 文本补全
    print("=== 文本补全 ===")
    result = llm("请介绍一下你自己")
    print(result)
    
    # 聊天对话
    print("\n=== 聊天对话 ===")
    messages = [
        SystemMessage(content="你是一个专业的AI助手"),
        HumanMessage(content="你好，请用简洁的语言介绍一下你自己")
    ]
    response = chat.generate([messages])
    print(response.generations[0][0].message.content)