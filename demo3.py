# 动态步骤
# backend.py
import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from dataclasses import dataclass
from langgraph.graph import StateGraph, START
from langchain.chat_models import init_chat_model

app = FastAPI()

# ---------------------------
# 模型初始化
# ---------------------------
llm = init_chat_model(model="openai:gpt-4o-mini")

# ---------------------------
# 状态定义
# ---------------------------
@dataclass
class ChatState:
    input_text: str
    steps: list = None
    result: str = ""

# ---------------------------
# 节点函数
# ---------------------------
async def router_node(state: ChatState):
    state.steps.append("[step1: Thinking]")
    await asyncio.sleep(0.2)  # 模拟处理延迟
    # 简单规则：根据关键字判断
    if "joke" in state.input_text.lower():
        task_type = "joke"
    elif "search" in state.input_text.lower():
        task_type = "rag"
    else:
        task_type = "chat"
    state.steps.append(f"[step1 done: Task identified: {task_type}]")
    return {"task_type": task_type}

async def llm_node(state: ChatState):
    state.steps.append("[step2: Generating response]")
    prompt = f"Respond to: {state.input_text}"
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    state.steps.append("[step2 done: Response generated]")
    state.result = response.content
    return {"result": state.result}

# ---------------------------
# 构建 LangGraph
# ---------------------------
graph = StateGraph(ChatState)
graph.add_node("router", router_node)
graph.add_node("llm", llm_node)
graph.add_edge(START, "router")
graph.add_edge("router", "llm")
compiled_graph = graph.compile()

# ---------------------------
# SSE 流式接口
# ---------------------------
@app.post("/chat_stream")
async def chat_stream(request: Request):
    data = await request.json()
    input_text = data.get("message", "")
    state = ChatState(input_text=input_text, steps=[])

    async def event_generator():
        async for chunk, metadata in compiled_graph.astream(state, stream_mode="messages"):
            # 推送步骤状态
            for step in getattr(chunk, "steps", []):
                yield f"data: {json.dumps({'type':'status','content':step})}\n\n"
            # 推送模型输出
            if getattr(chunk, "result", ""):
                yield f"data: {json.dumps({'type':'token','content':chunk.result})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    from IPython.display import Image, display 
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles 
    display(Image(graph.get_graph().draw_mermaid_png()))