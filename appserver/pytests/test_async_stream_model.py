import asyncio
import os
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.new_model import CustomChatModel
from langchain_core.messages import SystemMessage, HumanMessage

pytestmark = pytest.mark.asyncio

@pytest.mark.asyncio
async def test_async_stream():
    """测试异步流式输出"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("未设置DASHSCOPE_API_KEY环境变量")
    
    model = CustomChatModel()
    messages = [
        SystemMessage(content="你是一个专业的AI助手"),
        HumanMessage(content="请写一首关于春天的短诗")
    ]
    
    full_response = ""
    async for chunk in model._astream(messages):
        content = chunk.message.content
        full_response += str(content)
    
    assert full_response
    print(f"异步流式输出结果: {full_response}")

@pytest.mark.asyncio
async def test_async_stream_invoke():
    """测试异步流式astream方法"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("未设置DASHSCOPE_API_KEY环境变量")
    
    model = CustomChatModel()
    full_response = ""
    async for chunk in model.astream("请写一首关于春天的短诗"):
        content = chunk.content
        full_response += str(content)
    
    assert full_response
    print(f"异步流式astream结果: {full_response}")

@pytest.mark.asyncio
async def test_concurrent_async_stream():
    """测试并发异步流式输出"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("未设置DASHSCOPE_API_KEY环境变量")
    
    model = CustomChatModel()
    
    async def stream_task(prompt: str):
        full_response = ""
        async for chunk in model.astream(prompt):
            content = chunk.content
            full_response += str(content)
        return full_response
    
    prompts = [
        "请写一首关于春天的短诗",
        "请介绍一下人工智能的发展历史",
        "请解释什么是机器学习"
    ]
    
    tasks = [stream_task(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        assert result
    print(f"并发异步流式输出结果: {[r[:50] for r in results]}") 