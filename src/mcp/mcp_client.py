import asyncio
from typing import List
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    """Client for connecting to MCP servers (Tavily and DuckDuckGo)."""

    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

    def get_mcp_client(self, server_type: str) -> MultiServerMCPClient:
        """Create MCP client based on server type."""
        if server_type == "tavily":
            return MultiServerMCPClient({
                "tavily": {
                    "command": "docker",
                    "args": ["run", "-i", "--rm", "-e", "TAVILY_API_KEY", "mcp/tavily"],
                    "env": {"TAVILY_API_KEY": self.tavily_api_key},
                    "transport": "stdio"
                }
            })
        elif server_type == "duckduckgo":
            return MultiServerMCPClient({
                "duckduckgo": {
                    "command": "docker",
                    "args": ["run", "-i", "--rm", "mcp/duckduckgo"],
                    "transport": "stdio"
                }
            })
        else:
            raise ValueError(f"Unknown server type: {server_type}")

    async def get_tools_for_server(self, server_type: str) -> List:
        """
        Get ALL tools from specified MCP server.
        Let LangGraph/workflow decide which tool to use - no hardcoding.
        """
        client = self.get_mcp_client(server_type)
        try:
            tools = await client.get_tools()
            print(
                f"✓ Loaded {len(tools)} tools from {server_type}: {[t.name for t in tools]}")
            return tools
        except Exception as e:
            print(f"✗ Error getting tools from {server_type}: {e}")
            return []

    def filter_tools_by_name(self, tools: List, tool_names: List[str]) -> List:
        """
        Filter tools by exact names.
        Used for controlled experiments where you want specific tools only.
        """
        filtered = [tool for tool in tools if tool.name in tool_names]
        if filtered:
            print(
                f"✓ Filtered to {len(filtered)} tool(s): {[t.name for t in filtered]}")
        else:
            print(f"⚠ No tools matched names: {tool_names}")
        return filtered


# Tool name mappings for each server
SEARCH_TOOL_MAPPING = {
    "tavily": "tavily-search",      # Tavily's search tool
    "duckduckgo": "search"           # DuckDuckGo's search tool
}


async def get_search_tool_for_server(server_type: str):
    """
    Get the specific search tool for a given MCP server.
    This ensures we use the correct search tool per server for experiments.
    """
    mcp_client = MCPClient()

    try:
        # Load all tools from server
        all_tools = await mcp_client.get_tools_for_server(server_type)

        if not all_tools:
            return None

        # Get the search tool name for this server
        search_tool_name = SEARCH_TOOL_MAPPING.get(server_type)
        if not search_tool_name:
            print(f"⚠ No search tool mapping defined for {server_type}")
            return None

        # Filter to get only the search tool
        search_tools = mcp_client.filter_tools_by_name(
            all_tools, [search_tool_name])

        return search_tools[0] if search_tools else None

    except Exception as e:
        print(f"✗ Error getting search tool for {server_type}: {e}")
        return None
