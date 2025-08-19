"""
FastAPI + LangGraph + LangChain (DeepSeek) async streaming chatbot
-----------------------------------------------------------------------------
- Uses StateGraph to define workflow
- Asynchronous nodes + streaming via `app.astream_events`
- Threaded memory via MemorySaver + `thread_id`
- Loads environment variables from `.env` using python-dotenv

Run:
  uvicorn fastapi-langgraph-deepseek-chatbot:app_api --reload --port 8000

Test (SSE):
  curl -N -X POST "http://localhost:8000/stream" \
    -H "Content-Type: application/json" \
    -d '{"thread_id":"demo-1","message":"Tell me a joke about databases","language":"English"}'

Test (JSON one-shot):
  curl -s -X POST "http://localhost:8000/chat" \
    -H "Content-Type: application/json" \
    -d '{"thread_id":"demo-1","message":"Remember that my name is Todd.","language":"English"}' | jq

Notes:
- Place `DEEPSEEK_API_KEY=your_key_here` in a `.env` file or set it in the environment.
- Code style mirrors LangChain/LangGraph docs; all model calls are async.
"""

from typing import Any, AsyncGenerator, Dict, Optional
import asyncio
import json
import os
import getpass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Load .env into environment ---
config = load_dotenv()

# --- LangChain / LangGraph imports (follow docs style) ---
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Ensure API key is set (doc-style prompt if missing)
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter API key for DeepSeek: ")

# Initialize the DeepSeek chat model
model = init_chat_model("deepseek-chat", model_provider="deepseek")

# ----------------------------
# Graph definition (StateGraph)
# ----------------------------

# Optional prompt that enforces target language while preserving history
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Reply in {language}."),
        ("placeholder", "{messages}"),
    ]
)

# Async node that calls the model
async def call_model(state: MessagesState):
    language = state.get("language", "English")
    prompt = prompt_template.invoke({
        "messages": state["messages"],
        "language": language,
    })
    response = await model.ainvoke(prompt)
    # Merge by returning a dict with the messages key
    return {"messages": response}

# Build the graph
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add memory (thread-aware)
checkpointer = MemorySaver()
app_graph = workflow.compile(checkpointer=checkpointer)

# ----------------------------
# FastAPI app
# ----------------------------
app_api = FastAPI(title="DeepSeek LangGraph Chatbot", version="1.0")


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    language: Optional[str] = "English"


# Utility: build input message list from a user string
def make_input_messages(user_text: str) -> list[BaseMessage]:
    return [HumanMessage(user_text)]


# Async generator that streams tokens via LangGraph event stream
async def sse_token_stream(
    user_msg: str,
    language: str,
    thread_id: str,
) -> AsyncGenerator[bytes, None]:
    inputs: Dict[str, Any] = {
        "messages": make_input_messages(user_msg),
        "language": language,
    }
    config = {"configurable": {"thread_id": thread_id}}

    # astream_events yields granular events including token-level chunks
    async for event in app_graph.astream_events(inputs, config=config, version="v2"):
        et = event.get("event")
        print("event: ", et)
        if et == "on_chat_stream":
            # Token chunk emitted by the model
            data = event.get("data", {})
            chunk: AIMessage | BaseMessage | None = data.get("chunk")
            if chunk is None:
                continue
            # chunk.content is typically a string here
            token = getattr(chunk, "content", None)
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n".encode()
        elif et == "on_chain_end":
            # Final assembled message
            data = event.get("data", {})
            output = data.get("output")
            if output and isinstance(output, dict) and "messages" in output:
                messages = output["messages"]
                if messages and isinstance(messages, list):
                    final_text = messages[-1].content if hasattr(messages[-1], "content") else None
                    if final_text:
                        yield f"data: {json.dumps({'type': 'final', 'content': final_text})}\n\n".encode()
    # Signal completion
    yield b"data: [DONE]\n\n"


@app_api.post("/stream")
async def stream_chat(req: ChatRequest):
    async def gen():
        async for chunk in sse_token_stream(req.message, req.language or "English", req.thread_id):
            yield chunk
    return StreamingResponse(gen(), media_type="text/event-stream")


@app_api.post("/chat")
async def chat_once(req: ChatRequest):
    inputs = {"messages": make_input_messages(req.message), "language": req.language or "English"}
    config = {"configurable": {"thread_id": req.thread_id}}
    # Single async invocation (no streaming)
    output = await app_graph.ainvoke(inputs, config=config)
    # Extract last assistant message
    messages = output.get("messages", [])
    last = messages[-1].content if messages else ""
    return JSONResponse({"thread_id": req.thread_id, "message": last})


# Optional WebSocket streaming (convenient for browsers)
@app_api.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_json()
            message = payload.get("message", "")
            thread_id = payload.get("thread_id", "ws-default")
            language = payload.get("language", "English")

            inputs = {"messages": make_input_messages(message), "language": language}
            config = {"configurable": {"thread_id": thread_id}}

            async for event in app_graph.astream_events(inputs, config=config, version="v2"):
                et = event.get("event")
                if et == "on_chat_model_stream":
                    data = event.get("data", {})
                    chunk = data.get("chunk")
                    token = getattr(chunk, "content", None) if chunk else None
                    if token:
                        await ws.send_json({"type": "token", "content": token})
                elif et == "on_chain_end":
                    data = event.get("data", {})
                    output = data.get("output")
                    final_text: Optional[str] = None
                    if output and isinstance(output, dict) and "messages" in output:
                        msgs = output["messages"]
                        if msgs and isinstance(msgs, list):
                            final_text = msgs[-1].content if hasattr(msgs[-1], "content") else None
                    await ws.send_json({"type": "final", "content": final_text or ""})
            await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        return


# -------------
# Local testing
# -------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo:app_api", host="0.0.0.0", port=8000, reload=True)
