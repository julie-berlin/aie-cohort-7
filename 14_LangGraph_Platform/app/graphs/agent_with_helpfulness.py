"""An agent graph with a post-response helpfulness check loop.

After the agent responds, a secondary node evaluates helpfulness ('Y'/'N').
If helpful, end; otherwise, continue the loop or terminate after a safe limit.
"""
from __future__ import annotations

from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

from app.state import AgentState
from app.models import get_chat_model
from app.tools import get_tool_belt


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: AgentState) -> Dict[str, Any]:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_helpfulness(state: AgentState):
    """Decide whether to execute tools or run the helpfulness evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "helpfulness"


def helpfulness_node(state: AgentState) -> Dict[str, Any]:
    """Evaluate helpfulness of the latest response relative to the initial query."""
    # If we've exceeded loop limit, short-circuit with END decision marker
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="HELPFULNESS:END")]}    

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    prompt_template = """
  Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

  Initial Query:
  {initial_query}

  Final Response:
  {final_response}"""

    helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
    helpfulness_check_model = get_chat_model(model_name="gpt-4.1-mini")
    helpfulness_chain = (
        helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()
    )

    helpfulness_response = helpfulness_chain.invoke(
        {
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        }
    )

    decision = "Y" if "Y" in helpfulness_response else "N"
    return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}


def helpfulness_decision(state: AgentState):
    """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
    # Check loop-limit marker
    if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if "HELPFULNESS:Y" in text:
        return "end"
    return "continue"


def build_graph():
    """Build an agent graph with an auxiliary helpfulness evaluation subgraph."""
    graph = StateGraph(AgentState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()


