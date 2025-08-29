import asyncio
from typing import List, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


# --- 定义消息类型 ---
class MessagesState(TypedDict):
    messages: List[str]
    route: str


# --- 模拟 LLM 模型 ---
class DummyModel:
    async def ainvoke(self, prompt: str) -> str:
        await asyncio.sleep(1)  # 模拟调用耗时
        return f"[LLM 回复]: {prompt}"


model = DummyModel()


# --- 定义节点 ---
async def router_node(state: MessagesState):
    """路由节点：简单分类"""
    text = state["messages"][-1]
    if "summary" in text.lower():
        route = "summarizer"
    else:
        route = "model"
    return {"route": route}


async def model_node(state: MessagesState):
    """普通模型调用"""
    response = await model.ainvoke(state["messages"][-1])
    return {"messages": state["messages"] + [response]}


async def summarizer_node(state: MessagesState):
    """总结节点"""
    response = await model.ainvoke("请总结: " + state["messages"][-1])
    return {"messages": state["messages"] + [response]}


# --- 定义 workflow ---
workflow = StateGraph(MessagesState)

workflow.add_node("router", router_node)
workflow.add_node("model", model_node)
workflow.add_node("summarizer", summarizer_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["route"],  # 根据 route 字段选择下一个节点
    {"model": "model", "summarizer": "summarizer"},
)
workflow.add_edge("model", END)
workflow.add_edge("summarizer", END)

# 使用内存保存 checkpoint
app = workflow.compile()


# --- 运行 demo ---
async def main():
    input_state = {"messages": ["hello, I need a summary"], "route": ""}

    print(">>> 启动 workflow...")
    async for event in app.astream(input_state):
        for node, state in event.items():
            print(f"✅ 节点 {node} 执行完成, 当前状态: {state}")


if __name__ == "__main__":
    asyncio.run(main())
