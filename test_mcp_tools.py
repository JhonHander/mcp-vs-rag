"""
Test script to verify MCP tools connection and functionality.
Tests both Tavily and DuckDuckGo MCP servers running in Docker.
"""
import asyncio
from src.mcp.mcp_client import MCPClient, get_search_tool_for_server


async def test_mcp_connection(server_type: str):
    """Test basic MCP server connection and tool loading."""
    print(f"\n{'=' * 70}")
    print(f"Testing {server_type.upper()} MCP Server Connection")
    print('=' * 70)

    mcp_client = MCPClient()

    print(f"\n1. Attempting to connect to {server_type} server...")
    tools = await mcp_client.get_tools_for_server(server_type)

    if not tools:
        print(f"FAILED: Could not load tools from {server_type}")
        return False

    print(f"SUCCESS: Loaded {len(tools)} tools")
    print(f"Available tools: {[tool.name for tool in tools]}")

    return True


async def test_search_tool(server_type: str, query: str):
    """Test the specific search tool for a server."""
    print(f"\n{'=' * 70}")
    print(f"Testing {server_type.upper()} Search Tool")
    print('=' * 70)

    print(f"\nQuery: {query}")
    print("-" * 70)

    try:
        search_tool = await get_search_tool_for_server(server_type)

        if not search_tool:
            print(f"FAILED: Could not get search tool for {server_type}")
            return False

        print(f"Tool loaded: {search_tool.name}")
        print(f"Executing search...")

        result = await search_tool.ainvoke({"query": query})

        print(f"\nSUCCESS: Got results")
        print(f"Result type: {type(result)}")
        print(f"Result preview (first 300 chars):")
        print("-" * 70)
        result_str = str(result)
        print(result_str[:300] + "..." if len(result_str)
              > 300 else result_str)

        return True

    except Exception as e:
        print(f"FAILED: Error executing search")
        print(f"Error: {str(e)}")
        return False


async def test_both_servers():
    """Test both Tavily and DuckDuckGo servers."""
    test_query = "What are the latest developments in AI?"

    print("\n" + "=" * 70)
    print("MCP SERVERS VERIFICATION TEST")
    print("=" * 70)

    results = {}

    # Test Tavily
    tavily_connected = await test_mcp_connection("tavily")
    results['tavily_connection'] = tavily_connected

    if tavily_connected:
        tavily_search = await test_search_tool("tavily", test_query)
        results['tavily_search'] = tavily_search

    # Test DuckDuckGo
    duckduckgo_connected = await test_mcp_connection("duckduckgo")
    results['duckduckgo_connection'] = duckduckgo_connected

    if duckduckgo_connected:
        duckduckgo_search = await test_search_tool("duckduckgo", test_query)
        results['duckduckgo_search'] = duckduckgo_search

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED - MCP tools are working correctly")
    else:
        print("SOME TESTS FAILED - Check errors above")
    print("=" * 70)

    return all_passed


async def verify_docker_containers():
    """Verify that Docker is available and can run MCP containers."""
    import subprocess

    print("\n" + "=" * 70)
    print("DOCKER VERIFICATION")
    print("=" * 70)

    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"Docker version: {result.stdout.strip()}")

        print("\nChecking if MCP images are available...")
        images_result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=5
        )

        images = images_result.stdout.strip().split('\n')
        mcp_images = [img for img in images if 'mcp/' in img]

        if mcp_images:
            print(f"Found {len(mcp_images)} MCP images:")
            for img in mcp_images:
                print(f"  - {img}")
        else:
            print("No MCP images found locally")
            print("They will be pulled automatically on first use")

        return True

    except FileNotFoundError:
        print("FAILED: Docker is not installed or not in PATH")
        return False
    except Exception as e:
        print(f"FAILED: Error checking Docker: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# MCP TOOLS VERIFICATION SUITE")
    print("#" * 70)

    async def run_all_tests():
        docker_ok = await verify_docker_containers()

        if not docker_ok:
            print("\nCannot proceed without Docker")
            return

        await test_both_servers()

    asyncio.run(run_all_tests())
