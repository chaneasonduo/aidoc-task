#!/usr/bin/env python3
"""
测试异步流式输出功能
"""

import asyncio
import os
from new_model import CustomChatModel
from langchain_core.messages import SystemMessage, HumanMessage


async def test_async_stream():
    """测试异步流式输出"""
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        return
    
    try:
        # 创建模型实例
        model = CustomChatModel()
        print("✓ CustomChatModel创建成功")
        
        # 准备测试消息
        messages = [
            SystemMessage(content="你是一个专业的AI助手"),
            HumanMessage(content="请写一首关于春天的短诗")
        ]
        
        # 测试异步流式输出
        print("正在测试异步流式输出...")
        print("流式输出:")
        
        full_response = ""
        async for chunk in model._astream(messages):
            content = chunk.message.content
            print(content, end="", flush=True)
            full_response += str(content)
        
        print(f"\n\n✓ 异步流式输出测试成功")
        print(f"完整响应: {full_response}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


async def test_async_stream_invoke():
    """测试使用astream方法的异步流式输出"""
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        return
    
    try:
        # 创建模型实例
        model = CustomChatModel()
        print("✓ CustomChatModel创建成功")
        
        # 测试异步流式invoke
        print("正在测试异步流式astream...")
        print("流式输出:")
        
        full_response = ""
        async for chunk in model.astream("请写一首关于春天的短诗"):
            content = chunk.content
            print(content, end="", flush=True)
            full_response += str(content)
        
        print(f"\n\n✓ 异步流式astream测试成功")
        print(f"完整响应: {full_response}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


async def test_concurrent_async_stream():
    """测试并发异步流式输出"""
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        return
    
    try:
        # 创建模型实例
        model = CustomChatModel()
        print("✓ CustomChatModel创建成功")
        
        # 准备多个并发任务
        async def stream_task(prompt: str, task_id: int):
            print(f"\n任务{task_id}开始:")
            full_response = ""
            async for chunk in model.astream(prompt):
                content = chunk.content
                print(content, end="", flush=True)
                full_response += str(content)
            print(f"\n任务{task_id}完成: {full_response[:50]}...")
            return full_response
        
        prompts = [
            "请写一首关于春天的短诗",
            "请介绍一下人工智能的发展历史",
            "请解释什么是机器学习"
        ]
        
        print("正在测试并发异步流式输出...")
        tasks = [stream_task(prompt, i+1) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        
        print(f"\n✓ 并发异步流式输出测试成功")
        for i, result in enumerate(results):
            print(f"任务{i+1}结果: {result[:50]}...")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


async def main():
    """主测试函数"""
    print("=== 测试异步流式输出 ===")
    await test_async_stream()
    
    print("\n=== 测试异步流式astream ===")
    await test_async_stream_invoke()
    
    print("\n=== 测试并发异步流式输出 ===")
    await test_concurrent_async_stream()


if __name__ == "__main__":
    asyncio.run(main()) 