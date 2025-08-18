"""Simplified LangGraph A2A Client

A simple agent that queries another agent for business leadership information
then combines this with a prompt to help prepare for interviews.
"""

import asyncio
import logging
from typing import Annotated, Any, Dict, List, TypedDict
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleA2AState(TypedDict):
    """State for the simple A2A agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    agent_card: Any
    task_id: str | None
    context_id: str | None


async def fetch_agent_card():
    """Fetch the agent card from the A2A server"""
    base_url = 'http://localhost:10000'

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        logger.info(f'Fetching agent card from: {base_url}')
        agent_card = await resolver.get_agent_card()
        logger.info('Successfully fetched agent card')
        return agent_card


def extract_text_from_parts(parts) -> str | None:
    """Extract text content from a list of parts.

    Parts can be Pydantic models with a root attribute containing TextPart,
    or direct part objects with kind and text attributes.
    """
    for part in parts:
        # Part is a Pydantic model with a root attribute containing TextPart
        if hasattr(part, 'root'):
            part_root = part.root
            if hasattr(part_root, 'kind') and part_root.kind == 'text':
                if hasattr(part_root, 'text'):
                    return part_root.text
        # Fallback to direct attributes
        elif hasattr(part, 'kind') and part.kind == 'text':
            if hasattr(part, 'text'):
                return part.text
    return None


async def process_a2a_response(response_gen):
    """Process the streaming response from A2A agent.

    Returns a tuple of (message_content, task_id, context_id)
    """
    task_id = None
    context_id = None
    message_content = "No response content"

    async for chunk in response_gen:
        # chunk can be either (Task, Event) tuple or final Message
        if isinstance(chunk, tuple):
            # It's a task update
            task, event = chunk

            if task:
                task_id = task.id if hasattr(task, 'id') else None
                context_id = task.contextId if hasattr(task, 'contextId') else None

            if event:
                # Extract content from artifact
                if hasattr(event, 'artifact') and event.artifact:
                    artifact = event.artifact
                    if hasattr(artifact, 'parts'):
                        text = extract_text_from_parts(artifact.parts)
                        if text:
                            message_content = text

        elif isinstance(chunk, Message):
            # It's the final message response
            if hasattr(chunk, 'parts'):
                text = extract_text_from_parts(chunk.parts)
                if text:
                    message_content = text

    return message_content, task_id, context_id


async def query_a2a_agent(
    state: SimpleA2AState
) -> Dict[str, Any]:
    """Query the A2A agent with the user's message"""

    # Get the last human message
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"messages": [AIMessage(content="No user query found")]}

    # Get or fetch agent card
    agent_card = state.get("agent_card")
    if not agent_card:
        agent_card = await fetch_agent_card()

    # Create a Message object with proper camelCase fields
    message = Message(
        role='user',
        parts=[{'kind': 'text', 'text': user_message}],
        messageId=uuid4().hex,  # Note: camelCase
        contextId=state.get("context_id") if state.get("context_id") else None
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            # Create client configuration for JSON-RPC transport
            config = ClientConfig(
                httpx_client=httpx_client,
                supported_transports=['JSONRPC']  # Use uppercase to match agent card
            )

            # Create client factory and client
            factory = ClientFactory(config)
            client = factory.create(agent_card)

            logger.info(f"Sending query to A2A agent: {user_message[:50]}...")
            # ClientFactory's send_message returns an async generator for streaming
            response_gen = client.send_message(message)

            # Process the streaming response using helper function
            message_content, task_id, context_id = await process_a2a_response(response_gen)

            logger.info("Successfully received response from A2A agent")

            return {
                "messages": [AIMessage(content=message_content)],
                "agent_card": agent_card,
                "task_id": task_id,
                "context_id": context_id,
            }

    except Exception as e:
        logger.error(f"Error querying A2A agent: {e}")
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")],
            "agent_card": agent_card,
        }

def generate_interviewing_advice(state):
    """
    LangGraph node: Retrieves the last AIMessage from state['messages'] and generates interviewing advice.
    """
    messages = state.get("messages", [])
    last_ai_message = None
    # Find the last AIMessage in the messages list
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            last_ai_message = message
            break

    assert last_ai_message is not None, "No AIMessage found in state['messages']"

    ai_content = last_ai_message.content
    # Use the ai_content to generate interviewing advice via an LLM call

    # Define a prompt template for generating interviewing advice
    INTERVIEW_ADVICE_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert career coach. Given the following information from an AI assistant, provide actionable interviewing advice for a user preparing for a leadership interview. Be concise, specific, and practical.",
            ),
            ("human", "{ai_content}"),
        ]
    )

    async def get_interviewing_advice(ai_content: str) -> str:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        prompt = INTERVIEW_ADVICE_PROMPT.format(ai_content=ai_content)
        response = await llm.ainvoke(prompt)
        assert hasattr(response, "content"), "LLM response missing content"
        return response.content

    # Run the LLM call synchronously if not already in an event loop
    try:
        advice = asyncio.run(get_interviewing_advice(ai_content))
    except RuntimeError:
        # If already in an event loop (e.g., in Jupyter), use create_task
        advice = asyncio.get_event_loop().run_until_complete(get_interviewing_advice(ai_content))

    # Return new state with the advice as an AIMessage
    new_messages = messages + [AIMessage(content=advice)]
    return {**state, "messages": new_messages}



def create_simple_a2a_graph():
    """Create a simple LangGraph that queries the A2A agent"""

    graph = StateGraph(SimpleA2AState)

    # Single node that queries the A2A agent
    graph.add_node("query_a2a", query_a2a_agent)
    graph.add_node("generate_advice", generate_interviewing_advice)

    # Set entry and exit
    graph.set_entry_point("query_a2a")
    graph.add_edge("query_a2a", "generate_advice")
    graph.add_edge("generate_advice", END)

    return graph.compile()


async def demo_simple_a2a():
    """Demonstrate the simple A2A client"""

    # Create the graph
    graph = create_simple_a2a_graph()

    print("=" * 60)
    print("Simple LangGraph A2A Client Demo")
    print("=" * 60)

    # Example 1: Single query
    print("\n📌 Example 1: Single Query")
    print("-" * 40)

    query1 = "What is melioration in the context of leadership?"
    initial_state = {
        "messages": [HumanMessage(content=query1)],
        "agent_card": None,
        "task_id": None,
        "context_id": None,
    }

    print(f"User: {query1}")
    result = await graph.ainvoke(initial_state)

    # Display response
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\nAgent Response:\n{msg.content[:1000]}...")

    # Example 2: Multi-turn conversation
    print("\n\n📌 Example 2: Multi-turn Conversation")
    print("-" * 40)

    query2 = "What are the elements of the OODA Loop?"
    state2 = {
        "messages": [HumanMessage(content=query2)],
        "agent_card": result.get("agent_card"),  # Reuse agent card
        "task_id": None,
        "context_id": None,
    }

    print(f"User: {query2}")
    result2 = await graph.ainvoke(state2)

    for msg in result2["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\nAgent Response:\n{msg.content[:1000]}...")

    # Follow-up query with context
    if result2.get("task_id") and result2.get("context_id"):
        print("\n📌 Follow-up Query (with context)")
        print("-" * 40)

        follow_up = "Under what circumstances is OODA most useful?"
        state3 = {
            "messages": [HumanMessage(content=follow_up)],
            "agent_card": result2["agent_card"],
            "task_id": result2["task_id"],
            "context_id": result2["context_id"],
        }

        print(f"User: {follow_up}")
        result3 = await graph.ainvoke(state3)

        for msg in result3["messages"]:
            if isinstance(msg, AIMessage):
                print(f"\nAgent Response:\n{msg.content[:1000]}...")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_simple_a2a())
