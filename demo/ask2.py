from typing import Dict, List, Optional

from langchain.agents import Tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


# ---- 定义状态 ----
class State(BaseModel):
    input: str
    intent: Optional[str] = None
    retrieved_docs: Optional[List[Document]] = None
    tool_result: Optional[str] = None
    output: Optional[str] = None
    history: List[Dict[str, str]] = []


# ---- 初始化 LLM ----
llm = ChatDeepSeek(model="deepseek-chat", temperature=0, verbose=True)


# ---- Router 节点 ----
def router_node(state: State) -> Dict:
    question = state.input.lower()
    if any(k in question for k in ["冒烟", "压力"]):
        intent = "general"
    elif any(k in question for k in ["文档", "规范"]):
        intent = "rag"
    elif any(k in question for k in ["查询", "获取"]):
        intent = "function"
    else:
        intent = "react"
    print(f"[router] intent = {intent}")
    return {
        "intent": intent,
        "history": state.history + [{"role": "user", "content": state.input}],
    }


# ---- 通用问答节点 ----
def general_node(state: State) -> Dict:
    prompt = ChatPromptTemplate.from_messages(
        [("system", "你是测试管理专家，请回答问题"), ("user", "{question}")]
    )
    resp = llm.invoke({"question": state.input}).content
    print(f"[general] output = {resp}")
    history = state.history + [{"role": "assistant", "content": resp}]
    return {"output": resp, "history": history}


# ---- RAG 节点 ----
def mock_retriever(query: str) -> List[Document]:
    return [Document(page_content=f"知识库内容：关于 {query} 的说明")]


def rag_node(state: State) -> Dict:
    docs = mock_retriever(state.input)
    context = "\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "你是测试管理专家，请结合上下文回答"), ("user", "{question}")]
    )
    resp = llm.invoke({"question": state.input, "context": context}).content
    history = state.history + [{"role": "assistant", "content": resp}]
    print(f"[rag] output = {resp}")
    return {"output": resp, "retrieved_docs": docs, "history": history}


# ---- FunctionCall 节点 ----
def function_node(state: State) -> Dict:
    result = f"模拟查询结果: {state.input}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", "把工具结果转为自然语言输出"), ("user", "{tool_result}")]
    )
    resp = llm.invoke({"tool_result": result}).content
    history = state.history + [{"role": "assistant", "content": resp}]
    print(f"[function] output = {resp}")
    return {"output": resp, "tool_result": result, "history": history}


# ---- ReAct Agent 节点 ----
# 模拟工具
tools = [
    Tool(
        name="QueryDB", func=lambda q: f"数据库返回结果: {q}", description="查询数据库"
    ),
    Tool(name="SearchDocs", func=lambda q: f"检索到文档: {q}", description="文档搜索"),
]
react_agent = create_react_agent(model=llm, tools=tools)


def react_node(state: State) -> Dict:
    resp = react_agent.invoke(state.input)
    history = state.history + [{"role": "assistant", "content": resp}]
    print(f"[react] output = {resp}")
    return {"output": resp, "history": history}


# ---- 构建 Graph ----
graph = StateGraph(State)
graph.add_node("router", RunnableLambda(router_node))
graph.add_node("general", RunnableLambda(general_node))
graph.add_node("rag", RunnableLambda(rag_node))
graph.add_node("function", RunnableLambda(function_node))
graph.add_node("react", RunnableLambda(react_node))


def continue_router(state: State) -> str:
    if state.intent == "general":
        return "general"
    if state.intent == "rag":
        return "rag"
    if state.intent == "function":
        return "function"
    if state.intent == "react":
        return "react"
    return "general"


# edges
graph.add_conditional_edges(
    "router",
    path=continue_router,
    path_map={
        "general": "general",
        "rag": "rag",
        "function": "function",
        "react": "react",
    },
)

for n in ["general", "rag", "function", "react"]:
    graph.add_edge(n, END)

graph.set_entry_point("router")
workflow = graph.compile()

workflow.get_graph().draw_mermaid_png(output_file_path="graph-123.png")

# # ---- 测试 ----
# queries = [
#     "什么是冒烟测试？",  # general
#     "请结合文档解释压力测试",  # rag
#     "查询项目A最近一次测试结果",  # function
#     "帮我分析项目A和B的测试结果趋势",  # react
# ]

# for q in queries:
#     print("\n==============================")
#     print("用户问题:", q)
#     result = workflow.invoke({"input": q})
#     print("最终输出:", result["output"])
