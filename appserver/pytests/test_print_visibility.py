#!/usr/bin/env python3
"""
测试 print 可见性的简单脚本
"""

def test_print_visibility():
    """测试 print 是否可见"""
    print("=== 这是测试输出 ===")
    print("如果你能看到这些文字，说明配置生效了！")
    print("gen type: <class 'generator'>")
    assert True

def test_multiple_prints():
    """测试多个 print 语句"""
    print("第一个 print")
    print("第二个 print")
    print("第三个 print")
    assert True

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-s", "-v"]) 