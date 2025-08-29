import asyncio
from typing import Annotated, Literal, TypedDict

import dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages

dotenv.load_dotenv()


# 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]
    category: str


# LLM
llm = ChatDeepSeek(model="deepseek-chat")


class Category(TypedDict):
    category: Literal["greeting", "qa", "other"]


# Router 节点
async def router_node(state: State):
    structured_llm = llm.with_structured_output(Category)
    # 这里用非流式调用，直接拿最终结果
    async for ev in structured_llm.astream(state["messages"]):
        pass
    # 组装成最终对象
    resp = ev
    print(f"resp: {resp}")
    return {"category": resp["category"], "messages": state["messages"]}


# Greeting 节点（支持流式）
async def greeting_node(state: State):
    resp = await llm.ainvoke("你是一个热情的客服，给用户打个招呼。")
    return {"messages": [resp]}


# QA 节点（支持流式）
async def qa_node(state: State):
    resp = await llm.ainvoke("回答用户问题。")
    return {"messages": [resp]}


# 条件路由
def route(state: State):
    if state["category"] == "greeting":
        return "greeting"
    elif state["category"] == "qa":
        return "qa"
    else:
        return END


# 搭建 Graph
builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("greeting", greeting_node)
builder.add_node("qa", qa_node)

builder.add_conditional_edges("router", route)
builder.add_edge("greeting", END)
builder.add_edge("qa", END)
builder.add_edge(START, "router")

graph = builder.compile()


async def main():
    async for chunk, metadata in graph.astream(
        {"messages": ["你好"]}, stream_mode="messages"
    ):
        print("chunk: ", chunk)
        # print("metadata: ", metadata)


if __name__ == "__main__":
    asyncio.run(main())
