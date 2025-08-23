
# mitmdump -s proxy_script.py

from mitmproxy import http
import json
import pprint

def request(flow: http.HTTPFlow) -> None:
    # 过滤需要监控的域名（可根据实际情况调整）
    # if "smith.langchain.com" in flow.request.pretty_url or "deepseek.com" in flow.request.pretty_url:
    print("\n===== 捕获请求 =====")
    print(f"URL: {flow.request.pretty_url}")
    print(f"方法: {flow.request.method}")
    
    # 打印请求头
    print("\n请求头:")
    pprint.pprint(dict(flow.request.headers))
    
    # 打印请求体（根据内容类型解析）
    print("\n请求体:")
    try:
        # 尝试解析 JSON 格式
        if "application/json" in flow.request.headers.get("Content-Type", ""):
            data = json.loads(flow.request.text)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        # 处理 multipart 表单数据（常见于文件上传或复杂数据）
        elif "multipart/form-data" in flow.request.headers.get("Content-Type", ""):
            print("Multipart 数据:")
            for part in flow.request.multipart_form:
                print(f"  字段名: {part.name}")
                print(f"  内容类型: {part.headers.get('Content-Type', 'unknown')}")
                print(f"  内容长度: {len(part.content)} 字节")
                # 若内容是文本，可尝试解码
                try:
                    print(f"  内容: {part.content.decode('utf-8')[:500]}...")  # 只显示前500字符
                except:
                    print("  内容: 非文本数据")
        # 其他文本类型
        else:
            print(flow.request.text[:1000] + ("..." if len(flow.request.text) > 1000 else ""))
    except Exception as e:
        print(f"解析请求体失败: {str(e)}")
    print("====================\n")

def response(flow: http.HTTPFlow) -> None:
    # 过滤需要监控的域名
    # if "smith.langchain.com" in flow.request.pretty_url or "deepseek.com" in flow.request.pretty_url:
    print("\n===== 捕获响应 =====")
    print(f"URL: {flow.request.pretty_url}")
    print(f"状态码: {flow.response.status_code}")
    
    # 打印响应头
    print("\n响应头:")
    pprint.pprint(dict(flow.response.headers))
    
    # 打印响应体
    print("\n响应体:")
    try:
        if "application/json" in flow.response.headers.get("Content-Type", ""):
            data = json.loads(flow.response.text)
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(flow.response.text[:1000] + ("..." if len(flow.response.text) > 1000 else ""))
    except Exception as e:
        print(f"解析响应体失败: {str(e)}")
    print("====================\n")
