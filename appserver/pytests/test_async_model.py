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
async def test_async_generate():
    """测试异步生成功能"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("未设置DASHSCOPE_API_KEY环境变量")
    model = CustomChatModel()
    messages = [
        SystemMessage(content="你是一个专业的AI助手"),
        HumanMessage(content="你好，请用简洁的语言介绍一下你自己")
    ]
    result = await model._agenerate(messages)
    assert result.generations
    assert result.generations[0].message.content
    print(f"生成结果: {result.generations[0].message.content}")

@pytest.mark.asyncio
async def test_async_invoke():
    """测试异步invoke功能"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("未设置DASHSCOPE_API_KEY环境变量")
    model = CustomChatModel()
    result = await model.ainvoke("你好，请用简洁的语言介绍一下你自己")
    assert hasattr(result, "content")
    assert result.content
    print(f"调用结果: {result.content}")

@pytest.mark.asyncio
async def test_concurrent_calls():
    """测试并发调用"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("未设置DASHSCOPE_API_KEY环境变量")
    model = CustomChatModel()
    prompts = [
        "请写一首关于春天的短诗",
        "请介绍一下人工智能的发展历史",
        "请解释什么是机器学习"
    ]
    tasks = [model.ainvoke(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    for result in results:
        assert hasattr(result, "content")
        assert result.content
    print([result.content[:50] for result in results]) 