"""
Veld - Agentic Memory + LangChain Integration Example

This example shows how to use Veld - Agentic Memory as a persistent memory
backend for LangChain agents via the MCP adapter.

Requirements:
    pip install langchain langchain-mcp-adapters langchain-anthropic

Setup:
    1. Start shodh-memory server: npx -y @shodh/memory-mcp
    2. Or run locally: veld server
    3. Run this script

Documentation: https://github.com/Portll/veld/tree/main/docs
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent


async def main():
    """
    Create a LangChain agent with Veld - Agentic Memory for persistent context.

    The agent can:
    - Store memories with `remember`
    - Retrieve memories with `recall`
    - Get context automatically with `proactive_context`
    - Search by tags with `recall_by_tags`
    """

    # Configure Veld - Agentic Memory MCP server
    # Option 1: stdio transport (npx)
    async with MultiServerMCPClient(
        {
            "shodh-memory": {
                "command": "npx",
                "args": ["-y", "@shodh/memory-mcp"],
                "env": {
                    "SHODH_USER_ID": "langchain-agent",  # Optional: isolate memories per agent
                },
            }
        }
    ) as client:
        # Get tools from the MCP server
        tools = client.get_tools()

        print(f"Available Veld tools: {[t.name for t in tools]}")
        # Expected: ['remember', 'recall', 'proactive_context', 'recall_by_tags',
        #            'recall_by_date', 'forget', 'forget_by_tags', 'memory_stats',
        #            'context_summary', 'consolidation_report', 'verify_index', 'repair_index']

        # Create LangChain agent with Claude
        model = ChatAnthropic(model="claude-sonnet-4-20250514")
        agent = create_react_agent(model, tools)

        # Example 1: Store a memory
        print("\n--- Storing a memory ---")
        response = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": "Remember that the project uses PostgreSQL with pgvector for vector storage. "
                               "This is a Decision type memory with tags: database, architecture, project-x"
                }
            ]
        })
        print(response["messages"][-1].content)

        # Example 2: Recall memories
        print("\n--- Recalling memories ---")
        response = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": "What database does the project use? Search your memory."
                }
            ]
        })
        print(response["messages"][-1].content)

        # Example 3: Use proactive_context (recommended for every conversation start)
        print("\n--- Proactive context surfacing ---")
        response = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": "I want to add a new feature to the database layer. "
                               "First, use proactive_context to surface relevant memories about our database setup."
                }
            ]
        })
        print(response["messages"][-1].content)

        # Example 4: Get memory statistics
        print("\n--- Memory statistics ---")
        response = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": "Show me the memory system statistics using memory_stats."
                }
            ]
        })
        print(response["messages"][-1].content)


async def stateful_session_example():
    """
    Example with stateful MCP session for maintaining context across tool calls.

    Use this when you need the memory server to maintain state between calls
    (e.g., for transaction-like operations or complex workflows).
    """

    client = MultiServerMCPClient(
        {
            "shodh-memory": {
                "command": "npx",
                "args": ["-y", "@shodh/memory-mcp"],
            }
        }
    )

    # Create a persistent session
    async with client.session("shodh-memory") as session:
        tools = client.get_tools()
        model = ChatAnthropic(model="claude-sonnet-4-20250514")
        agent = create_react_agent(model, tools)

        # Multiple interactions share the same session
        for query in [
            "Remember: User prefers TypeScript over JavaScript",
            "Remember: User likes functional programming patterns",
            "What are the user's coding preferences? Check your memory.",
        ]:
            response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            print(f"\nQuery: {query}")
            print(f"Response: {response['messages'][-1].content[:200]}...")


async def http_transport_example():
    """
    Example using HTTP transport for a remote Veld server.

    Use this when running Veld - Agentic Memory as a standalone HTTP server
    rather than spawning via npx.
    """

    async with MultiServerMCPClient(
        {
            "shodh-memory": {
                "transport": "http",
                "url": "http://localhost:3030/mcp",  # Your Veld server URL
                "headers": {
                    "X-API-Key": "your-api-key",  # If authentication is enabled
                },
            }
        }
    ) as client:
        tools = client.get_tools()
        print(f"Connected to remote Veld server with {len(tools)} tools")

        # Use tools as normal...
        model = ChatAnthropic(model="claude-sonnet-4-20250514")
        agent = create_react_agent(model, tools)

        response = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What memories do you have? Use recall."}]
        })
        print(response["messages"][-1].content)


if __name__ == "__main__":
    print("=== Veld - Agentic Memory + LangChain Integration ===\n")
    asyncio.run(main())

    # Uncomment to try other examples:
    # asyncio.run(stateful_session_example())
    # asyncio.run(http_transport_example())
