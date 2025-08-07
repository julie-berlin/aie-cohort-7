"""Shared agent state definitions for LangGraph graphs."""
from __future__ import annotations

from typing import Annotated, TypedDict, List

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema for agent graphs, storing a message list with add_messages."""
    messages: Annotated[List, add_messages]


