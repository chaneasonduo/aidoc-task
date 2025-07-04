import json
import os
import sys
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

# 添加models目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from models.new_model import CustomChatModel, convert_message_to_dict


class TestConvertMessageToDict:
    """测试convert_message_to_dict函数"""
    
    def test_convert_system_message(self):
        """测试SystemMessage转换"""
        message = SystemMessage(content="你是一个AI助手")
        result = convert_message_to_dict(message)
        expected = {"role": "system", "content": "你是一个AI助手"}
        assert result == expected
    
    def test_convert_human_message(self):
        """测试HumanMessage转换"""
        message = HumanMessage(content="你好")
        result = convert_message_to_dict(message)
        expected = {"role": "user", "content": "你好"}
        assert result == expected
    
    def test_convert_ai_message(self):
        """测试AIMessage转换"""
        message = AIMessage(content="你好！我是AI助手")
        result = convert_message_to_dict(message)
        expected = {"role": "assistant", "content": "你好！我是AI助手"}
        assert result == expected
    
    def test_convert_unknown_message(self):
        """测试未知类型消息转换"""
        # 创建一个自定义消息类
        class CustomMessage(BaseMessage):
            def __init__(self, content):
                self.content = content
        
        message = CustomMessage(content="自定义消息")
        result = convert_message_to_dict(message)
        expected = {"role": "user", "content": "自定义消息"}
        assert result == expected
    
    def test_convert_empty_content(self):
        """测试空内容消息转换"""
        message = HumanMessage(content="")
        result = convert_message_to_dict(message)
        expected = {"role": "user", "content": ""}
        assert result == expected


class TestCustomChatModel:
    """测试CustomChatModel类"""
    
    @pytest.fixture
    def model(self):
        """创建测试用的模型实例"""
        return CustomChatModel(api_key="test_api_key_123")
    
    def test_init_with_api_key(self):
        """测试使用api_key初始化"""
        model = CustomChatModel(api_key="test_key")
        assert model.api_key == "test_key"
        assert model.api_url == "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    @patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'env_test_key'})
    def test_init_with_env_variable(self):
        """测试从环境变量获取api_key"""
        model = CustomChatModel()
        assert model.api_key == "env_test_key"
    
    def test_init_without_api_key_raises_error(self):
        """测试没有api_key时抛出错误"""
        # 临时清除环境变量
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as excinfo:
                CustomChatModel()
            assert "DASHSCOPE_API_KEY" in str(excinfo.value)
    
    @patch('new_model.requests.post')
    def test_generate_success(self, mock_post, model):
        """测试成功生成回复"""
        # 模拟API响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "text": "你好！我是一个AI助手，很高兴为您服务。"
            }
        }
        mock_post.return_value = mock_response
        
        # 测试消息
        messages = [
            SystemMessage(content="你是一个专业的AI助手"),
            HumanMessage(content="你好")
        ]
        
        # 执行生成
        result = model._generate(messages)
        
        # 验证结果
        assert result is not None
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "你好！我是一个AI助手，很高兴为您服务。"
        
        # 验证API调用
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['headers']['Authorization'] == f'Bearer {model.api_key}'
    
    @patch('new_model.requests.post')
    def test_generate_api_error(self, mock_post, model):
        """测试API错误处理"""
        # 模拟API错误
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response
        
        messages = [HumanMessage(content="测试")]
        
        # 验证抛出异常
        with pytest.raises(Exception) as excinfo:
            model._generate(messages)
        assert "API调用失败" in str(excinfo.value)


# 简单的函数测试，便于单独运行
def test_convert_system_message_simple():
    """简单的SystemMessage转换测试"""
    message = SystemMessage(content="测试消息")
    result = convert_message_to_dict(message)
    assert result["role"] == "system"
    assert result["content"] == "测试消息"


def test_convert_human_message_simple():
    """简单的HumanMessage转换测试"""
    message = HumanMessage(content="用户消息")
    result = convert_message_to_dict(message)
    assert result["role"] == "user"
    assert result["content"] == "用户消息"


def test_convert_ai_message_simple():
    """简单的AIMessage转换测试"""
    message = AIMessage(content="AI回复")
    result = convert_message_to_dict(message)
    assert result["role"] == "assistant"
    assert result["content"] == "AI回复" 