# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient

import os

# 配置代理（指向 mitmproxy 监听的端口）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:8080"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8080"

# 信任 mitmproxy 的 CA 证书（替换为你的证书路径）
# os.environ["REQUESTS_CA_BUNDLE"] = "~/.mitmproxy/mitmproxy-ca-cert.pem"  # Linux/macOS
os.environ["REQUESTS_CA_BUNDLE"] = "C:\\Users\\wpppp\\.mitmproxy\\mitmproxy-ca-cert.pem"  # Windows
os.environ["SSL_CERT_FILE"] = "C:\\Users\\wpppp\\.mitmproxy\\mitmproxy-ca-cert.pem"

# server_params = StdioServerParameters(
#     command="python",
#     # Make sure to update to the full absolute path to your math_server.py file
#     args=["math_mcp_server.py"],
# )

# server_params = StdioServerParameters(
#     command="python",
#     # Make sure to update to the full absolute path to your math_server.py file
#     args=["mcp-server/math_mcp_server.py"],
# )

async def async_main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["mcp-server/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # Make sure you start your weather server on port 8000
                "url": "http://localhost:8000/mcp/",
                "transport": "streamable_http",
            }
        }
    )
    model = ChatDeepSeek(model="deepseek-chat")
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    # math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})

    return weather_response


if __name__ == "__main__":
    import asyncio

    messages = asyncio.run(async_main())
    idx = 0
    for message in messages:

        print(f"message number {idx}", type(message), ":", message)
        idx+=1