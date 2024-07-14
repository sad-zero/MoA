import asyncio
import logging
from langchain.schema import HumanMessage
from langchain_core.runnables import Runnable
from mesop_langgraph.graph import GraphState, get_graph


graph: Runnable = get_graph()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    messages = [
        HumanMessage(content="How do I teach programming more funny?"),
    ]
    state: GraphState = {"depth": 0, "messages": messages}
    res: GraphState = asyncio.run(graph.ainvoke(state, debug=True))
    answer = res["messages"][-1].content
    logging.info(f"Answer: {answer}")
