from dataclasses import dataclass
from re import A
from typing import Annotated, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph, add_messages


class MyState(TypedDict):
    attitude: Literal["positive", "negative"]
    messages: Annotated[list, add_messages]


# llm = init_chat_model(model="openai:gpt-4o-mini")
class Attitude(TypedDict):
    attitude: Literal["positive", "negative"]


async def classify_attitude_node(state: MyState):
    llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0)

    new_llm = llm.with_structured_output(Attitude)
    llm_response = await new_llm.ainvoke(
        [
            {
                "role": "system",
                "content": "你是一个情感分析模型，你的任务是判断用户输入的句子是正面的还是负面的。",
            },
            {
                "role": "user",
                "content": f"判断以下句子是正面的还是负面的： {state['messages'][-1].content}",
            },
        ]
    )
    print(f"=====llm_response: {llm_response['attitude']}")
    return {"attitude": llm_response["attitude"]}


def condition_func(state: MyState):
    if state["attitude"] == "positive":
        return "positive"
    elif state["attitude"] == "negative":
        return "negative"
    else:
        return "other"


async def positive_node(state: MyState):
    # 再次请求 llm
    llm = ChatDeepSeek(model="deepseek-reasoner", temperature=1.5)
    llm_response = await llm.ainvoke(
        [
            {
                "role": "user",
                "content": f"写一首欢快的诗，不超过20个字",
            },
        ]
    )
    return {"messages": [{"role": "assistant", "content": llm_response.content}]}


async def negative_node(state: MyState):
    # 再次请求 llm
    llm = ChatDeepSeek(model="deepseek-reasoner", temperature=1.5)

    llm_response = await llm.ainvoke(
        [
            {
                "role": "user",
                "content": f"写一首悲伤的诗，不超过20个字",
            },
        ]
    )
    return {"messages": [{"role": "assistant", "content": llm_response.content}]}


graph = (
    StateGraph(MyState)
    .add_node("classify_attitude", classify_attitude_node)
    .add_node("positive", positive_node)
    .add_node("negative", negative_node)
    .add_edge(START, "classify_attitude")
    .add_conditional_edges(
        "classify_attitude",
        condition_func,
        path_map={"positive": "positive", "negative": "negative"},
    )
    .compile(checkpointer=InMemorySaver())
)


async def call_async():
    config = {"configurable": {"thread_id": "1"}}
    async for message_chunk, metadata in graph.astream(
        {"messages": [{"role": "user", "content": "我喜欢这个电影"}]},
        config=config,
        stream_mode="messages",
    ):
        # if message_chunk.content:
        #     print(message_chunk.content, end="|", flush=True)
        print(message_chunk.content)


import asyncio

asyncio.run(call_async())
