#!/usr/bin/env python3
"""
优化的流式生成测试
"""

import logging
from typing import List

import pytest
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_stream_generation_basic():
    """基础流式生成测试"""
    from models.new_model import CustomChatModel

    # 创建模型
    model = CustomChatModel()
    
    # 准备消息
    messages: List[BaseMessage] = [
        SystemMessage(content="你是一个AI助手"),
        HumanMessage(content="你好")
    ]
    
    # 执行流式生成
    result = model._stream(messages)
    
    # 基础验证
    assert result is not None, "流式生成结果不应为空"
    assert hasattr(result, '__iter__'), "结果应该是可迭代的"
    
    # 收集chunks
    chunks = list(result)
    
    # 验证chunks
    assert len(chunks) > 0, f"应该至少有一个chunk，实际: {len(chunks)}"
    
    # 验证每个chunk
    for i, chunk in enumerate(chunks):
        assert chunk is not None, f"chunk {i} 不应为空"
        logging.info(f"chunk {i+1}: {type(chunk)} - {chunk}")

def test_stream_generation_content():
    """测试流式生成的内容"""
    from models.new_model import CustomChatModel
    
    model = CustomChatModel()
    messages: List[BaseMessage] = [HumanMessage(content="请简单介绍一下Python")]
    
    result = model._stream(messages)
    chunks = []
    
    # 收集所有chunks
    for chunk in result:
        chunks.append(chunk)
        logging.info(f"收到chunk: {chunk}")
    
    # 验证内容
    assert len(chunks) > 0, "应该有chunks"
    
    # 合并内容
    total_content = ""
    for chunk in chunks:
        if hasattr(chunk, 'content'):
            total_content += chunk.content
        elif isinstance(chunk, str):
            total_content += chunk
        else:
            total_content += str(chunk)
    
    # 验证总内容
    assert len(total_content.strip()) > 0, "总内容不应为空"
    logging.info(f"总内容: {total_content[:100]}...")

def test_stream_generation_error_handling():
    """测试流式生成的错误处理"""
    from models.new_model import CustomChatModel
    
    model = CustomChatModel()
    
    # 测试空消息
    with pytest.raises(Exception):
        list(model._stream([]))

def test_stream_generation_performance():
    """测试流式生成的性能"""
    import time

    from models.new_model import CustomChatModel
    
    model = CustomChatModel()
    messages: List[BaseMessage] = [HumanMessage(content="请写一个简单的Python函数")]
    
    start_time = time.time()
    result = model._stream(messages)
    chunks = list(result)
    end_time = time.time()
    
    # 验证性能
    assert end_time - start_time < 30, "流式生成应在30秒内完成"
    assert len(chunks) > 0, "应该有chunks生成"
    
    logging.info(f"生成 {len(chunks)} 个chunks，耗时 {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"]) 