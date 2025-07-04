import os
import requests
import json

# DashScope官方API地址
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 从环境变量读取API KEY
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "qwen-plus",
    "input": {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"}
        ]
    },
    "parameters": {
        "result_format": "message"
    }
}

if __name__ == "__main__":
    print("请求DashScope API...\n")
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    print(f"状态码: {response.status_code}")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(response.text) 


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