import asyncio
import json
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from file_saver import save_to_file
from extract_messages import extract_messages

def load_dotenv():
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()


async def main():
    load_dotenv()
    MCP_SERVER = os.getenv("MCP_SERVER_PATH")
    model = init_chat_model("openai:gpt-4.1")

    client = MultiServerMCPClient(
        {
            "local-mcp-server": {
                "command": "python",
                "args": [MCP_SERVER],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()


    # Execute multiple queries and save structured outputs
    queries = [
        ("Roll 2d6", "dice_roll"),
        ("What is the exchange rate for British pounds to Euros?", "exchange_rate")
    ]
    
    for query, filename in queries:
        try:
            response = await graph.ainvoke({"messages": query})
            structured_output = extract_messages(response)
            save_to_file(json.dumps(structured_output, indent=2), filename)
            print(f"✅ Saved {filename} response")
        except Exception as error:
            print(f"❌ Error processing '{query}': {error}")


if __name__ == "__main__":
    asyncio.run(main())
