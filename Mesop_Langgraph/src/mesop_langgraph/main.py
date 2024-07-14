import asyncio
import logging
from typing import Annotated, List, TypedDict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph, add_messages
from langchain_community.chat_models.ollama import ChatOllama

LAYER_WIDTH, GRAPH_DEPTH = 3, 1

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    depth: int

async def aggregate_and_synthesize(state: GraphState):
    """
    Make an intermediate output
    Proposers and Aggregator use this prompt via state["messages"][-2:]
    """
    responses = state["messages"][-LAYER_WIDTH:]
    human_query = state["messages"][0].content
    aggregate_and_synthesize_prompt = """
You have been provided with a set of responses from various open-source models to the latest user query. Your
task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the
information provided in these responses, recognizing that some of it may be biased or incorrect. Your response
should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply
to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of
accuracy and reliability.
Responses from models:
"""
    for idx, response in enumerate(responses):
        aggregate_and_synthesize_prompt += f"{idx+1}: {response.content}\n"
    messages = [
        SystemMessage(content=aggregate_and_synthesize_prompt),
        HumanMessage(content=human_query),
    ]
    return {"messages": messages}

async def depth_checker(state: GraphState):
    """
    Determine whether the process ends.
    """
    if state["depth"] == GRAPH_DEPTH:
        return "aggregator"
    else:
        return "entry"
    
async def entry(state: GraphState):
    """
    Add graph depth
    """
    return {"depth": state["depth"] + 1}

async def proposer1(state: GraphState):
    system_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Act as a helpful assistant"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatOllama(model="gemma:2b-instruct", temperature=0.2, max_tokens=1024)
    model = system_template | llm

    input_ = state["messages"][-2:]
    response = model.invoke({"messages": input_})
    return {"messages": response}


async def proposer2(state: GraphState):
    system_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Act as a helpful assistant"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatOllama(model="qwen2:1.5b-instruct-fp16", temperature=0.2, max_tokens=1024)
    model = system_template | llm
    input_ = state["messages"][-2:]
    response = model.invoke({"messages": input_})
    return {"messages": response}


async def proposer3(state: GraphState):
    system_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Act as a helpful assistant"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatOllama(model="phi3:3.8b-instruct", temperature=0.2, max_tokens=1024)
    model = system_template | llm
    input_ = state["messages"][-2:]
    response = model.invoke({"messages": input_})
    return {"messages": response}

async def aggregator(state: GraphState):
    system_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Act as a helpful assistant"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatOllama(model="qwen2:7b-instruct", temperature=0.2, max_tokens=1024)
    model = system_template | llm
    input_ = state["messages"][-2:]
    response = model.invoke({"messages": input_})
    return {"messages": response}


async def main():
    graph_builder = StateGraph(GraphState)
    proposers = {
        "proposer1": proposer1,
        "proposer2": proposer2,
        "proposer3": proposer3,
    }
    # Add nodes
    graph_builder.add_node("entry", entry)
    for name, node in proposers.items():
        graph_builder.add_node(name, node)
    graph_builder.add_node("asp_node", aggregate_and_synthesize)
    graph_builder.add_node("aggregator", aggregator)
    # Add edges
    graph_builder.add_edge(START, "entry")
    for proposer in proposers.keys():
        graph_builder.add_edge("entry", proposer)
        graph_builder.add_edge(proposer, "asp_node")
    
    graph_builder.add_conditional_edges("asp_node", depth_checker)
    graph_builder.add_edge("aggregator", END)
    graph = graph_builder.compile()

    # print(graph.get_graph().draw_mermaid())
    messages = [
        HumanMessage(content="How do I teach programming more funny?"),
    ]
    state: GraphState = {"depth": 0, "messages": messages}
    res = await graph.ainvoke(state, debug=True)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    res: MessagesState = asyncio.run(main())
    answer = res["messages"][-1].content
    logging.info(f"Answer: {answer}")