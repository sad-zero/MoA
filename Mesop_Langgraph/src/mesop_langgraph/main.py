import asyncio
import logging
from langchain.schema import HumanMessage
from langchain_core.runnables import Runnable
from mesop_langgraph.graph import GraphState, get_graph
import mesop as me
import mesop.labs as mel

logging.basicConfig(level=logging.DEBUG)
graph: Runnable = get_graph()


@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/chat",
    title="Mesop Demo Chat",
)
def page():
    mel.chat(transform, title="Mesop Demo Chat", bot_user="Mesop Bot")


def transform(input: str, history: list[mel.ChatMessage]):
    messages = [
        HumanMessage(content=input),
    ]
    state: GraphState = {"depth": 0, "messages": messages}
    response: GraphState = asyncio.run(graph.ainvoke(state, debug=True))
    for intermediate_output in response["intermediate_outputs"]:
        for model, output in intermediate_output.items():
            yield f"{model}: {output}\n\n"
    yield f"Final Response: {response['messages'][-1].content}"
