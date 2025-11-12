"""
Configuration settings for MCP (Model Context Protocol) search tools.

This module centralizes all MCP-related configuration to avoid excessive
token usage and control costs.
"""

# Maximum number of search results to retrieve from each MCP server
MCP_SEARCH_LIMITS = {
    "tavily": {
        "max_results": 3,  # Number of search results (default is usually 5-10)
        "search_depth": "basic"  # "basic" or "advanced" - basic returns shorter results
    },
    "duckduckgo": {
        "max_results": 3  # Number of search results
    }
}

# Maximum character length for MCP context before truncation
# This prevents extremely long contexts that consume too many tokens
MAX_CONTEXT_LENGTH = 3000  # characters (~750 tokens approximately)

# Cost thresholds for warnings (in USD)
COST_WARNING_THRESHOLDS = {
    "per_experiment": 0.10,  # Warn if single experiment exceeds $0.10
    "per_batch": 1.00  # Warn if full batch exceeds $1.00
}

# RAG context limits (for consistency)
RAG_SEARCH_LIMITS = {
    "max_chunks": 5,  # Maximum chunks to retrieve from vector DB
    "max_chunk_length": 500  # Maximum characters per chunk
}


def get_mcp_search_config(server_type: str) -> dict:
    """
    Get search configuration for a specific MCP server.
    
    Args:
        server_type: Type of MCP server ("tavily" or "duckduckgo")
        
    Returns:
        Dictionary with search parameters for that server
    """
    return MCP_SEARCH_LIMITS.get(server_type, {"max_results": 3})


def should_truncate_context(context: str) -> tuple[bool, str]:
    """
    Check if context should be truncated and return truncated version.
    
    Args:
        context: The context string to check
        
    Returns:
        Tuple of (should_truncate: bool, truncated_context: str)
    """
    if len(context) > MAX_CONTEXT_LENGTH:
        truncated = context[:MAX_CONTEXT_LENGTH] + "\n\n[Context truncated to avoid excessive token usage]"
        return True, truncated
    return False, context
