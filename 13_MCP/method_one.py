import os
import asyncio

# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

def load_dotenv():
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()

async def main():
    load_dotenv()
    MCP_SERVER = os.getenv("MCP_SERVER_PATH")
    server_params = StdioServerParameters(
        command="python",
        args=[MCP_SERVER],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent("openai:gpt-4.1", tools)
            # Example query with error handling
            try:
                agent_response = await agent.ainvoke({
                    "messages": "What is the exchange rate for dollars to renminbi?"
                })
                print("✅ Agent Response:")
                print(agent_response)
            except Exception as error:
                print(f"❌ Error: {error}")

if __name__ == "__main__":
    asyncio.run(main())
