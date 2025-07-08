#!/usr/bin/env python3
"""
测试环境变量加载
"""
import os

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env 文件加载成功")
except ImportError:
    print("❌ python-dotenv 未安装")
    exit(1)

# 检查环境变量
print("\n环境变量检查:")
print(f"ANTHROPIC_API_KEY: {'✅ 已设置' if os.getenv('ANTHROPIC_API_KEY') else '❌ 未设置'}")
print(f"COHERE_API_KEY: {'✅ 已设置' if os.getenv('COHERE_API_KEY') else '❌ 未设置'}")
print(f"OPENAI_API_KEY: {'✅ 已设置' if os.getenv('OPENAI_API_KEY') else '❌ 未设置'}")
print(f"EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL', '使用默认值')}")

if __name__ == "__main__":
    print("\n🎉 环境变量配置完成！") 