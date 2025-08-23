from mitmproxy import http
import json
import pprint
from datetime import datetime
import os

# 日志文件路径
LOG_FILE = "langchain_mcp_interactions.log"

# 确保日志文件存在
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("LangChain与MCP服务交互日志（包含本地MCP）\n")
        f.write("=============================\n\n")

# 计数器，用于标记请求/响应顺序
sequence_counter = 1

def log_to_file(content):
    """将内容写入日志文件"""
    global sequence_counter
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] 序号: {sequence_counter}\n")
        f.write(content)
        f.write("\n" + "="*80 + "\n\n")
    
    sequence_counter += 1

def format_request(flow: http.HTTPFlow) -> str:
    """格式化请求信息"""
    parts = []
    parts.append(f"请求 URL: {flow.request.pretty_url}")
    parts.append(f"请求方法: {flow.request.method}")
    
    parts.append("\n请求头:")
    parts.append(pprint.pformat(dict(flow.request.headers)))
    
    parts.append("\n请求体:")
    try:
        if "application/json" in flow.request.headers.get("Content-Type", ""):
            data = json.loads(flow.request.text)
            parts.append(json.dumps(data, indent=2, ensure_ascii=False))
        elif "multipart/form-data" in flow.request.headers.get("Content-Type", ""):
            parts.append("Multipart 数据:")
            for part in flow.request.multipart_form:
                parts.append(f"  字段名: {part.name}")
                parts.append(f"  内容类型: {part.headers.get('Content-Type', 'unknown')}")
                parts.append(f"  内容长度: {len(part.content)} 字节")
                try:
                    part_content = part.content.decode('utf-8')
                    if len(part_content) > 500:
                        part_content = part_content[:500] + "..."
                    parts.append(f"  内容: {part_content}")
                except:
                    parts.append("  内容: 非文本数据")
        else:
            if flow.request.text:
                display_text = flow.request.text[:1000]
                if len(flow.request.text) > 1000:
                    display_text += "..."
                parts.append(display_text)
            else:
                parts.append("(无请求体)")
    except Exception as e:
        parts.append(f"解析请求体失败: {str(e)}")
    
    return "\n".join(parts)

def format_response(flow: http.HTTPFlow) -> str:
    """格式化响应信息"""
    parts = []
    parts.append(f"响应 URL: {flow.request.pretty_url}")
    parts.append(f"状态码: {flow.response.status_code}")
    
    parts.append("\n响应头:")
    parts.append(pprint.pformat(dict(flow.response.headers)))
    
    parts.append("\n响应体:")
    try:
        if "application/json" in flow.response.headers.get("Content-Type", ""):
            data = json.loads(flow.response.text)
            parts.append(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            if flow.response.text:
                display_text = flow.response.text[:1000]
                if len(flow.response.text) > 1000:
                    display_text += "..."
                parts.append(display_text)
            else:
                parts.append("(无响应体)")
    except Exception as e:
        parts.append(f"解析响应体失败: {str(e)}")
    
    return "\n".join(parts)

def should_capture(flow: http.HTTPFlow) -> bool:
    """判断是否需要捕获该请求（包含 MCP 服务、LangChain、DeepSeek）"""
    url = flow.request.pretty_url
    # 监控本地 MCP 服务（根据实际端口修改，这里包含 8000、8080 等常见端口）
    mcp_local = "localhost" in url and any(f":{port}" in url for port in [8000, 8080, 9000, 5000])
    # 保留原有监控的服务
    other_services = "smith.langchain.com" in url or "deepseek.com" in url
    return mcp_local or other_services

def request(flow: http.HTTPFlow) -> None:
    if should_capture(flow):
        print("\n===== 捕获请求 =====")
        request_str = format_request(flow)
        print(request_str)
        print("====================\n")
        log_to_file(f"[请求]\n{request_str}")

def response(flow: http.HTTPFlow) -> None:
    if should_capture(flow):
        print("\n===== 捕获响应 =====")
        response_str = format_response(flow)
        print(response_str)
        print("====================\n")
        log_to_file(f"[响应]\n{response_str}")
    