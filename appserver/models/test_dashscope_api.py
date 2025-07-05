#!/usr/bin/env python3
"""
测试DashScope API客户端的功能
"""

import os
from new_model import DashScopeAPIClient
from langchain_core.messages import SystemMessage, HumanMessage


def test_dashscope_api_client():
    """测试DashScope API客户端"""
    
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
            HumanMessage(content="你好，请用简洁的语言介绍一下你自己")
        ]
        
        # 调用API
        print("正在调用DashScope API...")
        response = client.call_api(
            messages=messages,
            model_name="qwen-turbo",
            temperature=0.7,
            max_tokens=100
        )
        
        print("✓ API调用成功")
        print(f"响应内容: {response}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


def test_custom_chat_model():
    """测试CustomChatModel"""
    
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量，跳过模型测试")
        return
    
    try:
        from new_model import CustomChatModel
        
        # 创建模型实例
        model = CustomChatModel()
        print("✓ CustomChatModel创建成功")
        
        # 测试_generate方法
        messages = [
            SystemMessage(content="你是一个专业的AI助手"),
            HumanMessage(content="你好，请用简洁的语言介绍一下你自己")
        ]
        
        print("正在测试_generate方法...")
        result = model._generate(messages)
        print("✓ _generate方法测试成功")
        print(f"生成结果: {result}")
        
        # 测试invoke方法
        print("正在测试invoke方法...")
        result = model.invoke("你好，请用简洁的语言介绍一下你自己")
        print("✓ invoke方法测试成功")
        print(f"调用结果: {result}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")


if __name__ == "__main__":
    print("=== 测试DashScope API客户端 ===")
    test_dashscope_api_client()
    
    print("\n=== 测试CustomChatModel ===")
    test_custom_chat_model()


#import os
# import dashscope

# messages = [
#     {'role': 'system', 'content': 'You are a helpful assistant.'},
#     {'role': 'user', 'content': '你是谁？'}
#     ]
# response = dashscope.Generation.call(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key=os.getenv('DASHSCOPE_API_KEY'),
#     model="qwen-plus", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     messages=messages,
#     result_format='message'
#     )
# print(response)