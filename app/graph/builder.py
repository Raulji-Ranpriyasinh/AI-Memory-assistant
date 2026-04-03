"""
Graph builder — assembles the LangGraph StateGraph from node functions.
"""

from __future__ import annotations

from langgraph.graph import END, START, MessagesState, StateGraph

from app.graph.nodes import make_nodes
from app.memory.controller import MemoryController


def build_graph(controller: MemoryController) -> StateGraph:
    """
    Build and return an *uncompiled* StateGraph.
    Compile it with checkpointer + store at call time (see chatbot.py).
    """
    nodes = make_nodes(controller)

    builder = StateGraph(MessagesState)

    builder.add_node("stm_a_update",  nodes["stm_a_update"])
    builder.add_node("stm_b_update",  nodes["stm_b_update"])
    builder.add_node("stm_c_extract", nodes["stm_c_extract"])
    builder.add_node("ltm_gate",      nodes["ltm_gate"])
    builder.add_node("chat",          nodes["chat"])

    builder.add_edge(START,           "stm_a_update")
    builder.add_edge("stm_a_update",  "stm_b_update")
    builder.add_edge("stm_b_update",  "stm_c_extract")
    builder.add_edge("stm_c_extract", "ltm_gate")
    builder.add_edge("ltm_gate",      "chat")
    builder.add_edge("chat",           END)

    return builder
