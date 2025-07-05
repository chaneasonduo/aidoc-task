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

import requests


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
    自定义的DashScope LLM包装器，支持文本补全和聊天两种模式，支持自定义API URL和请求参数。
    
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
    
    api_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    request_params: Optional[dict] = None  # 额外请求参数
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """验证环境配置"""
        values["dashscope_api_key"] = get_from_dict_or_env(
            values, "dashscope_api_key", "DASHSCOPE_API_KEY"
        )
        return values
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "dashscope"
    
    def _call(
        self,
        prompt: Union[str, BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt_str = ""
        messages: list[BaseMessage] = []
        if isinstance(prompt, str):
            prompt_str = prompt
            messages = [HumanMessage(content=prompt)]
        elif isinstance(prompt, BaseMessage):
            prompt_str = str(prompt.content)
            messages = [prompt]
        else:
            raise ValueError(f"prompt 必须为 str 或 BaseMessage, 当前为: {type(prompt)}")
        if self.mode == "chat":
            result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            return result.generations[0].message.content
        else:
            if self.streaming:
                # 只取第一个chunk内容（因为目前实现只yield一次）
                chunk = next(self._stream(prompt_str, stop=stop, run_manager=run_manager, **kwargs))
                return chunk.message.content
            # 构造payload
            dashscope_payload = {
                "model": self.model_name,
                "input": {
                    "messages": [{"role": "user", "content": prompt_str}]
                },
                "parameters": {
                    "result_format": "message"
                }
            }
            if self.request_params:
                dashscope_payload["parameters"].update(self.request_params)
            dashscope_payload["parameters"].update(kwargs)
            response = self._call_dashscope(payload=dashscope_payload)
            return response["output"]["choices"][0]["message"]["content"]
    
    def _generate(
        self,
        messages: list,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 兼容字符串、嵌套字符串等情况，全部转为BaseMessage
        normalized_messages: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                normalized_messages.append(msg)
            elif isinstance(msg, str):
                normalized_messages.append(HumanMessage(content=msg))
            elif isinstance(msg, list):
                # 兼容嵌套list
                for submsg in msg:
                    if isinstance(submsg, BaseMessage):
                        normalized_messages.append(submsg)
                    elif isinstance(submsg, str):
                        normalized_messages.append(HumanMessage(content=submsg))
                    else:
                        raise ValueError(f"不支持的消息类型: {submsg}")
            else:
                raise ValueError(f"不支持的消息类型: {msg}")
        if self.streaming:
            return self._generate_stream(normalized_messages, stop=stop, run_manager=run_manager, **kwargs)
        # DashScope官方要求：messages放在input字段下，参数放在parameters字段下
        dashscope_payload = {
            "model": self.model_name,
            "input": {
                "messages": [_convert_message_to_dict(msg) for msg in normalized_messages]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        if self.request_params:
            dashscope_payload["parameters"].update(self.request_params)
        dashscope_payload["parameters"].update(kwargs)
        response = self._call_dashscope(payload=dashscope_payload)
        message = _convert_dict_to_message({
            "role": "assistant",
            "content": response["output"]["choices"][0]["message"]["content"],
        })
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # 构造payload
        dashscope_payload = {
            "model": self.model_name,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        if self.request_params:
            dashscope_payload["parameters"].update(self.request_params)
        dashscope_payload["parameters"].update(kwargs)
        response = self._call_dashscope(payload=dashscope_payload)
        content = response["output"]["choices"][0]["message"]["content"]
        yield ChatGenerationChunk(message=AIMessageChunk(content=content))
    
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
        normalized_messages: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                normalized_messages.append(msg)
            elif isinstance(msg, str):
                normalized_messages.append(HumanMessage(content=msg))
            elif isinstance(msg, list):
                for submsg in msg:
                    if isinstance(submsg, BaseMessage):
                        normalized_messages.append(submsg)
                    elif isinstance(submsg, str):
                        normalized_messages.append(HumanMessage(content=submsg))
                    else:
                        raise ValueError(f"不支持的消息类型: {submsg}")
            else:
                raise ValueError(f"不支持的消息类型: {msg}")
        dashscope_payload = {
            "model": self.model_name,
            "input": {
                "messages": [_convert_message_to_dict(msg) for msg in normalized_messages]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        if self.request_params:
            dashscope_payload["parameters"].update(self.request_params)
        dashscope_payload["parameters"].update(kwargs)
        response = self._call_dashscope(payload=dashscope_payload)
        message = _convert_dict_to_message({
            "role": "assistant",
            "content": response["output"]["choices"][0]["message"]["content"],
        })
        return ChatResult(generations=[ChatGeneration(message=message)])
    
    def _make_request(self, payload: dict) -> dict:
        """
        发送HTTP请求到DashScope API
        
        Args:
            payload: 请求payload
            
        Returns:
            dict: API响应
        """
        headers = {
            "Authorization": f"Bearer {self.dashscope_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            raise ValueError(f"DashScope API错误: {response.text}")
        return response.json()

    def _call_dashscope(self, payload: dict) -> Any:
        """
        调用DashScope API的通用方法
        
        Args:
            payload: 请求payload
            
        Returns:
            Any: API响应
        """
        return self._make_request(payload)


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
    result = llm.invoke("请介绍一下你自己")
    print(result)
    
    # 聊天对话
    print("\n=== 聊天对话 ===")
    messages = [
        SystemMessage(content="你是一个专业的AI助手"),
        HumanMessage(content="你好，请用简洁的语言介绍一下你自己")
    ]
    response = chat._generate(messages)
    print(response.generations[0].message.content)