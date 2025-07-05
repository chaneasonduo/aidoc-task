#!/usr/bin/env python3
"""
调试流式API响应
"""

import os
import json
import requests
from new_model import DashScopeAPIClient
from langchain_core.messages import SystemMessage, HumanMessage


def debug_stream_api():
    """调试流式API响应"""
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        return
    
    # 创建API客户端
    client = DashScopeAPIClient(api_key=api_key)
    
    # 准备测试消息
    messages = [
        SystemMessage(content="你是一个专业的AI助手"),
        HumanMessage(content="请写一首关于春天的短诗")
    ]
    
    # 构建请求数据
    payload = {
        "model": "qwen-turbo",
        "input": {
            "messages": [{"role": "system", "content": "你是一个专业的AI助手"}, 
                        {"role": "user", "content": "请写一首关于春天的短诗"}]
        },
        "parameters": {
            "result_format": "message",
            "temperature": 0.7,
            "max_tokens": 200,
            "incremental_output": True
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    print("发送请求...")
    print(f"URL: {client.api_url}")
    print(f"Headers: {headers}")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    # 发送请求
    response = requests.post(
        client.api_url,
        json=payload,
        headers=headers,
        timeout=60,
        stream=True
    )
    
    print(f"状态码: {response.status_code}")
    print(f"响应头: {dict(response.headers)}")
    
    if response.status_code != 200:
        print(f"错误响应: {response.text}")
        return
    
    print("\n开始处理流式响应:")
    chunk_count = 0
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            print(f"原始行: {line}")
            
            if line.startswith('data: '):
                data = line[6:]
                try:
                    chunk_data = json.loads(data)
                    print(f"解析的JSON: {json.dumps(chunk_data, ensure_ascii=False, indent=2)}")
                    
                    if 'output' in chunk_data and 'choices' in chunk_data['output']:
                        choice = chunk_data['output']['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            if content:
                                chunk_count += 1
                                print(f"✓ 第{chunk_count}个chunk: '{content}'")
                            else:
                                print("空内容")
                        else:
                            print("没有找到message.content")
                    else:
                        print("没有找到output.choices")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    print(f"原始数据: {data}")
    
    print(f"\n总共处理了 {chunk_count} 个chunk")


if __name__ == "__main__":
    debug_stream_api() 