"""
Graph builder — assembles the LangGraph StateGraph from node functions.

Phase 4 update: Added health_correlation node with conditional routing
between ltm_gate and chat.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from app.graph.nodes import make_nodes
from app.memory.controller import MemoryController
from app.config.settings import ENABLE_CGM


def build_graph(controller: MemoryController) -> StateGraph:
    """
    Build and return an *uncompiled* StateGraph.
    Compile it with checkpointer + store at call time (see chatbot.py).

    Pipeline flow:
        START → stm_a_update → stm_b_update → stm_c_extract → ltm_gate
            → [conditional: health_correlation OR chat]
            → health_correlation → chat → END
    """
    nodes = make_nodes(controller)

    builder = StateGraph(MessagesState)

    builder.add_node("stm_a_update",       nodes["stm_a_update"])
    builder.add_node("stm_b_update",       nodes["stm_b_update"])
    builder.add_node("stm_c_extract",      nodes["stm_c_extract"])
    builder.add_node("ltm_gate",           nodes["ltm_gate"])
    builder.add_node("health_correlation", nodes["health_correlation"])
    builder.add_node("chat",               nodes["chat"])

    builder.add_edge(START,           "stm_a_update")
    builder.add_edge("stm_a_update",  "stm_b_update")
    builder.add_edge("stm_b_update",  "stm_c_extract")
    builder.add_edge("stm_c_extract", "ltm_gate")

    # Conditional routing: ltm_gate → health_correlation (if CGM enabled + health keywords) OR → chat
    builder.add_conditional_edges("ltm_gate", _route_after_ltm)
    builder.add_edge("health_correlation", "chat")
    builder.add_edge("chat",               END)

    return builder


def _route_after_ltm(state: MessagesState) -> Literal["health_correlation", "chat"]:
    """
    Conditional edge function.
    Checks ENABLE_CGM flag and health keywords in the latest message.
    Returns 'health_correlation' if CGM is enabled and message contains health keywords.
    Returns 'chat' otherwise.
    """
    if not ENABLE_CGM:
        return "chat"

    HEALTH_KEYWORDS = {
        'glucose', 'cgm', 'sugar', 'ate', 'meal', 'food',
        'mood', 'stress', 'exercise', 'walk', 'run',
    }

    messages = state.get("messages", [])
    latest_user_msg = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )

    if latest_user_msg:
        content_lower = latest_user_msg.content.lower()
        if any(kw in content_lower for kw in HEALTH_KEYWORDS):
            return "health_correlation"

    return "chat"
