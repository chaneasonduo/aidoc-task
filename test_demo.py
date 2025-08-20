"""
Python API 测试脚本 (requests 版)
--------------------
- 调用 /chat 和 /stream API
- 使用 requests（同步）
- 可直接运行：python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_chat():
    """测试一次性 JSON 返回的 /chat API"""
    payload = {
        "thread_id": "test-thread-1",
        "message": "Hi, I'm Bob",
        "language": "English",
    }
    resp = requests.post(f"{BASE_URL}/chat", json=payload)
    print("Status:", resp.status_code)
    try:
        print("Response JSON:", resp.json())
    except Exception:
        print("Raw Response:", resp.text)


def test_stream():
    """测试流式 /stream API (SSE)"""
    payload = {
        "thread_id": "test-thread-2",
        "message": "What's my name?",
        "language": "English",
    }
    with requests.post(f"{BASE_URL}/stream", json=payload, stream=True) as resp:
        print("Status:", resp.status_code)
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    print("\n--- Stream finished ---")
                    break
                try:
                    msg = json.loads(data)
                    if msg["type"] == "token":
                        print(msg["content"], end="", flush=True)
                    elif msg["type"] == "final":
                        print(f"\n[Final] {msg['content']}")
                except json.JSONDecodeError:
                    print(f"[Raw] {data}")


def test_stream_v2():
    """测试流式 /stream/v2 API (SSE)"""
    payload = {
        "thread_id": "test-thread-2",
        "message": "What's my name?",
        "language": "English",
    }
    with requests.post(f"{BASE_URL}/stream/v2", json=payload, stream=True) as resp:
        print("Status:", resp.status_code)
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    print("\n--- Stream finished ---")
                    break
                try:
                    msg = json.loads(data)
                    if msg["type"] == "token":
                        print(msg["content"], end="", flush=True)
                    elif msg["type"] == "final":
                        print(f"\n[Final] {msg['content']}")
                except json.JSONDecodeError:
                    print(f"[Raw] {data}")

def main():
    # test_chat()
    print("===================")
    test_stream_v2()



if __name__ == "__main__":
    main()
