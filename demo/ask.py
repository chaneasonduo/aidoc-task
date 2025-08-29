from typing import Annotated, Any, Dict, Literal, Optional

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.globals import set_verbose
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

load_dotenv()
set_verbose(True)
# ---- 定义LLM ----
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


# ---- 状态定义 ----
class State(BaseModel):
    input: str
    intent: Optional[Literal["general", "rag", "function"]] = None
    retrieved_docs: Optional[list[Document]] = None
    tool_result: Optional[Any] = None
    output: Optional[str] = None


# ---- 路由节点 ----
def router_node(state: State) -> Dict:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个问题分类器，负责将用户问题分类为以下三类："
                "1. general 通用问题（如解释概念、聊天等）"
                "2. rag 专业问题，需要从知识库检索"
                "3. function 操作类问题，需要调用工具执行",
            ),
            ("user", "{question}"),
        ]
    )
    chain = prompt | llm
    resp = str(chain.invoke({"question": state.input}).content).lower()
    if "rag" in resp:
        intent = "rag"
    elif "function" in resp:
        intent = "function"
    else:
        intent = "general"
    print("intent:", intent)
    return {"intent": intent}


# ---- 通用问答节点 ----
def general_node(state: State) -> Dict:
    print("general_node")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是测试管理领域的专家，请直接回答用户的问题。"),
            ("user", "{question}"),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke({"question": state.input})
    return {"output": resp.content}


# ---- RAG 节点 (检索 + 生成) ----
# 模拟retriever
def mock_retriever(query: str) -> list[Document]:
    print("mock_retriever")
    return [Document(page_content=f"知识库中关于 {query} 的说明...")]


def rag_node(state: State) -> Dict:
    print("rag_node")
    docs = mock_retriever(state.input)
    context = "\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是测试管理领域的专家。结合提供的上下文回答问题：\n{context}"),
            ("user", "{question}"),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke({"context": context, "question": state.input})
    return {"retrieved_docs": docs, "output": resp.content}


# ---- Function Call 节点 ----
def mock_tool(query: str) -> str:
    print("mock_tool")
    return f"查询结果：关于 {query} 的一些测试数据"


def function_node(state: State) -> Dict:
    print("function_node")
    tool_result = mock_tool(state.input)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是测试管理助手，请把工具返回的结果转为自然语言输出。"),
            ("user", "{tool_result}"),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke({"tool_result": tool_result})
    return {"tool_result": tool_result, "output": resp.content}


# ---- 构建Graph ----
workflow = StateGraph(state_schema=State)

workflow.add_node("router", RunnableLambda(router_node))
workflow.add_node("general", RunnableLambda(general_node))
workflow.add_node("rag", RunnableLambda(rag_node))
workflow.add_node("function", RunnableLambda(function_node))


def continue_router(state: State) -> str:
    if state.intent == "general":
        return "general"
    if state.intent == "rag":
        return "rag"
    if state.intent == "function":
        return "function"
    return "general"


# edges
workflow.add_conditional_edges(
    source="router",
    path=continue_router,
    path_map={"general": "general", "rag": "rag", "function": "function"},
)

workflow.add_edge("general", END)
workflow.add_edge("rag", END)
workflow.add_edge("function", END)

# entrypoint
workflow.set_entry_point("router")

graph = workflow.compile()


# ---- 测试 ----
query = "帮我查询一下“手机银行系统”有多少条缺陷"

# png_bytes = graph.get_graph().draw_mermaid_png()
# with open("workflow.png", "wb") as f:
#     f.write(png_bytes)
# print("流程图已保存到 workflow.png")

result = graph.invoke({"input": query})
print("输出:", result["output"])

# query = "请解释在微服务测试中如何进行压力测试？"
# result = graph.invoke({"input": query})
# print("输出:", result["output"])

# query = "查询项目A最近一次冒烟测试结果"
# result = graph.invoke({"input": query})
# print("输出:", result["output"])
