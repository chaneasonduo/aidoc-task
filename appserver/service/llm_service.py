import os
from typing import AsyncGenerator, List, Optional, Union

from langchain_community.chat_models import Tongyi
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class LLMService:
    """
    LLM服务封装类，提供统一的LLM调用接口
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2000,
        **kwargs
    ):
        """
        初始化LLM服务
        
        Args:
            model_name: 模型名称，默认为gpt-3.5-turbo
            temperature: 温度参数，控制生成文本的随机性
            max_tokens: 最大生成长度
            **kwargs: 其他模型参数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = self._init_model(**kwargs)
    
    def _init_model(self, **kwargs) -> BaseChatModel:
        """
        根据模型名称初始化对应的ChatModel
        """
        model_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        
        # 根据模型名称选择不同的模型
        if "qwen" in self.model_name.lower():
            return Tongyi(
                model_name=self.model_name,
                **model_kwargs
            )
        else:
            # 默认使用OpenAI
             return Tongyi(
                model_name=self.model_name,
                **model_kwargs
            )
    
    async def chat(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> str:
        """
        同步聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他模型参数
            
        Returns:
            模型生成的回复
        """
        response = await self.model.agenerate([messages], **kwargs)
        return response.generations[0][0].text
    
    async def achat(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        异步流式聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他模型参数
            
        Yields:
            模型生成的回复片段
        """
        async for chunk in self.model.astream(messages, **kwargs):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)


# 全局实例
llm_service = LLMService()
