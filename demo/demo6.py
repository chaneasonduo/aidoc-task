from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import START, StateGraph


@dataclass
class MyState:
    topic: str
    joke: str = ""


# llm = init_chat_model(model="openai:gpt-4o-mini")
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


def call_model(state: MyState):
    """Call the LLM to generate a joke about a topic"""
    llm_response = llm.invoke(
        [{"role": "user", "content": f"Generate a joke about {state.topic}"}]
    )
    print(f"llm_response: {llm_response}")
    return {"joke": llm_response.content}


graph = StateGraph(MyState).add_node(call_model).add_edge(START, "call_model").compile()

for message_chunk, metadata in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)
