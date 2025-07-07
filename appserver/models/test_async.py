#!/usr/bin/env python3
"""
测试异步功能
"""

import asyncio
import os
from new_model import CustomChatModel
from langchain_core.messages import SystemMessage, HumanMessage


async def test_async_generate():
    """测试异步生成功能"""
    
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
            HumanMessage(content="你好，请用简洁的语言介绍一下你自己")
        ]
        
        # 测试异步生成
        print("正在测试异步生成...")
        result = await model._agenerate(messages)
        print("✓ 异步生成测试成功")
        print(f"生成结果: {result}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


async def test_async_invoke():
    """测试异步invoke功能"""
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        return
    
    try:
        # 创建模型实例
        model = CustomChatModel()
        print("✓ CustomChatModel创建成功")
        
        # 测试异步invoke
        print("正在测试异步invoke...")
        result = await model.ainvoke("你好，请用简洁的语言介绍一下你自己")
        print("✓ 异步invoke测试成功")
        print(f"调用结果: {result}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


async def test_concurrent_calls():
    """测试并发调用"""
    
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
        tasks = []
        prompts = [
            "请写一首关于春天的短诗",
            "请介绍一下人工智能的发展历史",
            "请解释什么是机器学习"
        ]
        
        print("正在测试并发调用...")
        for i, prompt in enumerate(prompts):
            task = model.ainvoke(prompt)
            tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks)
        
        print("✓ 并发调用测试成功")
        for i, result in enumerate(results):
            print(f"任务{i+1}: {result.content[:50]}...")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


async def main():
    """主测试函数"""
    print("=== 测试异步生成 ===")
    await test_async_generate()
    
    print("\n=== 测试异步invoke ===")
    await test_async_invoke()
    
    print("\n=== 测试并发调用 ===")
    await test_concurrent_calls()


if __name__ == "__main__":
    asyncio.run(main()) 