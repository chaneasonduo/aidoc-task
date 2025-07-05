#!/usr/bin/env python3
"""
测试流式输出功能
"""

import os
from new_model import CustomChatModel, DashScopeAPIClient
from langchain_core.messages import SystemMessage, HumanMessage


def test_stream_api_client():
    """测试API客户端的流式输出"""
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量，跳过API测试")
        return
    
    try:
        # 创建API客户端
        client = DashScopeAPIClient(api_key=api_key)
        print("✓ API客户端创建成功")
        
        # 准备测试消息
        messages = [
            SystemMessage(content="你是一个专业的AI助手"),
            HumanMessage(content="请写一首关于春天的短诗")
        ]
        
        # 调用流式API
        print("正在调用流式DashScope API...")
        print("流式输出:")
        
        full_response = ""
        headers = client._build_headers()
        headers['X-DashScope-Stream'] = 'true'
        for chunk in client.call_api_stream(
            messages=messages,
            model_name="qwen-turbo",
            temperature=0.7,
            max_tokens=200
        ):
            print(chunk, end="", flush=True)
            full_response += str(chunk)
        
        print(f"\n\n✓ 流式API调用成功")
        print(f"完整响应: {full_response}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


def test_stream_chat_model():
    """测试CustomChatModel的流式输出"""
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量，跳过模型测试")
        return
    
    try:
        # 创建模型实例
        model = CustomChatModel()
        print("✓ CustomChatModel创建成功")
        
        # 测试流式输出
        messages = [
            SystemMessage(content="你是一个专业的AI助手"),
            HumanMessage(content="请写一首关于春天的短诗")
        ]
        
        print("正在测试流式输出...")
        print("流式输出:")
        
        full_response = ""
        for chunk in model._stream(messages):
            content = chunk.message.content
            print(content, end="", flush=True)
            full_response += str(content)
        
        print(f"\n\n✓ 流式输出测试成功")
        print(f"完整响应: {full_response}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


def test_stream_invoke():
    """测试使用invoke方法的流式输出"""
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量，跳过测试")
        return
    
    try:
        # 创建模型实例
        model = CustomChatModel()
        print("✓ CustomChatModel创建成功")
        
        # 测试流式invoke
        print("正在测试流式invoke...")
        print("流式输出:")
        
        full_response = ""
        for chunk in model.stream("请写一首关于春天的短诗"):
            content = chunk.content
            print(content, end="", flush=True)
            full_response += str(content)
        
        print(f"\n\n✓ 流式invoke测试成功")
        print(f"完整响应: {full_response}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


if __name__ == "__main__":
    print("=== 测试API客户端流式输出 ===")
    test_stream_api_client()
    
    print("\n=== 测试CustomChatModel流式输出 ===")
    test_stream_chat_model()
    
    print("\n=== 测试流式invoke ===")
    test_stream_invoke() 