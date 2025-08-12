"""Agent graph with a post-response helpfulness check loop for A2A protocol compatibility.

After the agent responds, a secondary node evaluates helpfulness ('Y'/'N').
If helpful, end; otherwise, continue the loop or terminate after a safe limit.
"""
from __future__ import annotations

from typing import Dict, Any, Annotated, TypedDict, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage


class AgentState(TypedDict):
    """State schema for agent graphs, storing a message list with add_messages."""
    messages: Annotated[List, add_messages]
    structured_response: Any  # ResponseFormat | None


def build_model_with_tools(model):
    """Return a model instance bound to the tool belt."""
    from app.tools import get_tool_belt
    return model.bind_tools(get_tool_belt())


def call_model(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Invoke the model with the accumulated messages and append its response."""
    model_with_tools = build_model_with_tools(model)
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_helpfulness(state: Dict[str, Any]):
    """Decide whether to execute tools or run the helpfulness evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "helpfulness"


def helpfulness_node(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Evaluate helpfulness of the latest response relative to the initial query."""
    # If we've exceeded loop limit, short-circuit with END decision marker
    if len(state["messages"]) > 10:
        return {"messages": [AIMessage(content="HELPFULNESS:END")]}    

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    prompt_template = """
  Given an initial query and a final response, determine if the final response is extremely helpful or not. 
  A helpful response should:
  - Provide accurate and relevant information
  - Be complete and address the user's specific need
  - Use appropriate tools when necessary
  
  Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

  Initial Query:
  {initial_query}

  Final Response:
  {final_response}"""

    helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
    helpfulness_chain = (
        helpfulness_prompt_template | model | StrOutputParser()
    )

    helpfulness_response = helpfulness_chain.invoke(
        {
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        }
    )

    decision = "Y" if "Y" in helpfulness_response else "N"
    return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}


def helpfulness_decision(state: Dict[str, Any]):
    """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
    # Check loop-limit marker
    if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if "HELPFULNESS:Y" in text:
        return "end"
    return "continue"


def build_agent_graph_with_helpfulness(model, system_instruction, format_instruction, checkpointer=None):
    """Build an agent graph with an auxiliary helpfulness evaluation subgraph."""
    from app.tools import get_tool_belt
    from app.agent import ResponseFormat
    
    # Create model-bound functions
    def _call_model(state: AgentState) -> Dict[str, Any]:
        """Wrapper to pass model to call_model."""
        model_with_tools = build_model_with_tools(model)
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        
        # If there are no tool calls, try to extract structured response
        if not getattr(response, "tool_calls", None):
            try:
                # Apply response format to the model
                model_with_format = model.with_structured_output(
                    ResponseFormat,
                    method="json_schema",
                    include_raw=False
                )
                
                # Add system and format instructions
                formatted_messages = [("system", f"{system_instruction}\n\n{format_instruction}")] + state["messages"]
                structured_response = model_with_format.invoke(formatted_messages)
                
                return {
                    "messages": [response],
                    "structured_response": structured_response
                }
            except:
                # If structured output fails, just return the response
                return {"messages": [response]}
        else:
            # If there are tool calls, just return the response
            return {"messages": [response]}
    
    def _helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Wrapper to pass model to helpfulness_node."""
        return helpfulness_node(state, model)
    
    graph = StateGraph(AgentState)
    tool_node = ToolNode(get_tool_belt())
    
    graph.add_node("agent", _call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", _helpfulness_node)
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
    
    return graph.compile(checkpointer=checkpointer)
