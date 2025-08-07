"""Toolbelt assembly for agents.

Collects third-party tools and local tools (like RAG) into a single list that
graphs can bind to their language models.
"""
from __future__ import annotations

from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from app.rag import retrieve_information


def get_tool_belt() -> List:
    """Return the list of tools available to agents (Tavily, Arxiv, RAG)."""
    tavily_tool = TavilySearchResults(max_results=5)
    return [tavily_tool, ArxivQueryRun(), retrieve_information]


