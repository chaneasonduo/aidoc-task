# app.py
import asyncio
import json
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START

# ---------------------------
# 1️⃣ 定义状态
# ---------------------------
@dataclass
class MyState:
    topic: str
    joke: str = ""

# ---------------------------
# 2️⃣ 初始化 LLM
# ---------------------------
llm_joke = init_chat_model("deepseek-chat", model_provider="deepseek", tags=["llm_joke"])
llm_poem = init_chat_model("deepseek-chat", model_provider="deepseek", tags=["poem"])

# ---------------------------
# 3️⃣ 定义模型调用函数
# ---------------------------
async def call_joke_model(state: MyState):
    response = await llm_joke.ainvoke(
        [{"role": "user", "content": f"Generate a joke about {state.topic}"}]
    )
    return {"joke": response.content}

async def call_poem_model(state: MyState):
    response = await llm_poem.ainvoke(
        [{"role": "user", "content": f"Write a short poem about {state.topic}"}]
    )
    return {"poem": response.content}

# ---------------------------
# 4️⃣ 构建 Graph
# ---------------------------
graph = (
    StateGraph(MyState)
    .add_node("joke_node", call_joke_model)
    .add_node("poem_node", call_poem_model)
    .add_edge(START, "joke_node")
    .add_edge("joke_node", "poem_node")
    .compile()
)

# ---------------------------
# 5️⃣ FastAPI 应用
# ---------------------------
app = FastAPI()

@app.post("/stream")
async def stream_endpoint(request: Request):
    data = await request.json()
    topic = data.get("topic", "cats")

    async def event_generator():
        async for msg, metadata in graph.astream(
            {"topic": topic},
            stream_mode="messages",
        ):
            # token / chunk 输出
            yield f"data: {json.dumps({'type':'token','content': msg.content})}\n\n"
        # 流结束
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ---------------------------
# 6️⃣ 简单同步 /chat API（非流式）
# ---------------------------
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    topic = data.get("topic", "cats")
    # 异步调用 Graph，直接获取最终输出
    result = await graph.ainvoke({"topic": topic})
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo2:app", host="0.0.0.0", port=8000, reload=True)
