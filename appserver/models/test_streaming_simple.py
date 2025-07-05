#!/usr/bin/env python3
"""
简单测试流式输出功能
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from new_model import CustomChatModel


def test_streaming():
    """测试流式输出功能"""
    print("=== 测试流式输出功能 ===")
    
    # 创建模型实例
    model = CustomChatModel(
        model_name="qwen-turbo",
        temperature=0.7,
        max_tokens=1000
    )
    
    # 准备测试消息
    messages = [
        SystemMessage(content="你是一个专业的AI助手"),
        HumanMessage(content="你好，请用简洁的语言介绍一下你自己")
    ]
    
    print("开始流式输出...")
    print("=" * 50)
    
    # 测试流式输出
    try:
        for i, chunk in enumerate(model._stream(messages)):
            print(f"Chunk {i+1}: {chunk.message.content}")
        print("=" * 50)
        print("流式输出完成")
        return True
    except Exception as e:
        print(f"流式输出测试失败: {e}")
        return False


if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误: 请设置DASHSCOPE_API_KEY环境变量")
        sys.exit(1)
    
    # 运行测试
    if test_streaming():
        print("\n✅ 流式输出测试通过!")
    else:
        print("\n❌ 流式输出测试失败!")
        sys.exit(1) 