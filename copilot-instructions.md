# GitHub Copilot Instructions - RAG vs MCP Comparison Project

## Project Overview

This project implements a comparison system between static search (RAG - Retrieval Augmented Generation) and dynamic search (MCP - Model Context Protocol) to evaluate the effectiveness of different information retrieval pipelines in AI applications.

### General Architecture

**Parallel Execution with Unified Output**:

```
                    User Prompt
                         ↓
          ┌──────────────┴──────────────┐
          ↓                             ↓
    RAG Branch                     MCP Branch
    (Retrieve → Generate → Eval)   (Search → Generate → Eval)
          ↓                             ↓
          └──────────────┬──────────────┘
                         ↓
                  Merge Results
                         ↓
              Single Unified JSON
              (Contains both RAG and MCP results)
```

**Important**: 
- RAG and MCP execute **in parallel** (using LangGraph)
- Each branch maintains its **own context** (never mixed)
- Both results are **merged into ONE JSON** for comparison
- Single execution produces unified output with both evaluations

## Main Technology Stack

- **Orchestration Framework**: LangGraph
- **Vector Database**: Qdrant (Docker)
- **MCP Servers**: Tavily and DuckDuckGo (both in Docker)
- **Evaluation Framework**: RAGAS (metrics: Answer Relevancy and Faithfulness)
- **LLM Models**: GPT-5 (OpenAI, released August 2025) and Gemini 2.5 Pro (Google)
- **Language**: Python

## System Components

### 1. RAG Pipeline (already existing)

The project has two types of RAG:
- **NaiveRAG**: Basic RAG implementation
- **HybridRAG**: Advanced RAG implementation with hybrid search

**Important Note**: The chunking, indexing, and embedding creation processes are already complete. Only the new Qdrant database configuration in Docker is required, along with a folder to store chunks and embeddings.

### 2. Qdrant Configuration

**Instructions for Copilot**:
- Check the official Qdrant documentation at: https://qdrant.tech/documentation/
- For LangChain integration, refer to: https://qdrant.tech/documentation/frameworks/langchain/
- Use the MCP server `context7` to access updated Qdrant documentation
- The connection must be configured for Qdrant running in Docker
- Use `langchain-qdrant` as the integration library

**Qdrant Docker Configuration**:
```yaml
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

### 3. MCP Servers Configuration

Both MCP servers (Tavily and DuckDuckGo) will be running in Docker.

**MCP Docker Configuration**:
```json
{
  "mcp_docker": {
    "transport": "stdio",
    "command": "docker",
    "args": ["mcp", "gateway", "run"]
  }
}
```

**Instructions for Copilot**:
- Check Tavily MCP documentation on Docker Hub and official repositories
- Check DuckDuckGo MCP server documentation
- Use the MCP server `context7` to access updated documentation
- Both servers must be consumable from the same workflow
- Only one MCP server must run per execution (see execution matrix)

### 4. Routing and Tool Selection

**Challenge**: Implement a mechanism that allows executing only one of the two MCP servers (Tavily or DuckDuckGo) per execution.

**Solution**:
- Programmatic tool filtering: Implement conditional logic to filter available tools based on workflow configuration.
- LangGraph with conditional edges: Use `add_conditional_edges` from a routing node to direct flow to the specific tool (Tavily or DuckDuckGo).
- Parameterized execution function: Create a function that receives parameters (model, RAG type, MCP server) and configures the graph dynamically with the selected tool.

**Instructions for Copilot**:
- Check LangGraph documentation on conditional routing and tool selection
- Check the MCP server `context7` for updated LangChain documentation
- Verify in the official `docs by langchain` the best practices for:
  - Dynamic tool selection
  - Tool filtering in agents
  - Conditional routing in workflows
- Prioritize the most efficient and maintainable solution

### 5. Evaluation with RAGAS

**Metrics to implement**:
- **Answer Relevancy**: Measures how relevant the generated answer is to the question
- **Faithfulness**: Measures how faithful the answer is to the retrieved context (avoids hallucinations)

**Instructions for Copilot**:
- Check official RAGAS documentation at: https://docs.ragas.io/
- Implement only these two metrics (not all available)
- The evaluation result must include numerical scores for both metrics

### 6. Output Format

**Each configuration produces ONE unified JSON file containing BOTH RAG and MCP results:**

#### Unified Result (e.g., `gpt5_naive_tavily_1.json`)
```json
{
  "execution_id": "string",
  "timestamp": "ISO 8601 datetime",
  "configuration": {
    "model": "gpt-5 | gemini-2.5-pro",
    "rag_type": "naive | hybrid",
    "mcp_server": "tavily | duckduckgo"
  },
  "prompt": "string",
  
  "rag_results": {
    "retrieved_context": ["string"],
    "generated_answer": "string",
    "ragas_metrics": {
      "answer_relevancy": float,
      "faithfulness": float
    }
  },
  
  "mcp_results": {
    "retrieved_context": ["string"],
    "generated_answer": "string",
    "ragas_metrics": {
      "answer_relevancy": float,
      "faithfulness": float
    }
  }
}
```

## Execution Matrix

The system produces **8 unified JSON files**, each containing both RAG and MCP results:

### With GPT-5:
1. **Config 1**: `Prompt → GPT-5 → [NaiveRAG || Tavily] → Unified JSON`
2. **Config 2**: `Prompt → GPT-5 → [NaiveRAG || DuckDuckGo] → Unified JSON`
3. **Config 3**: `Prompt → GPT-5 → [HybridRAG || Tavily] → Unified JSON`
4. **Config 4**: `Prompt → GPT-5 → [HybridRAG || DuckDuckGo] → Unified JSON`

### With Gemini 2.5 Pro:
5. **Config 5**: `Prompt → Gemini-2.5-Pro → [NaiveRAG || Tavily] → Unified JSON`
6. **Config 6**: `Prompt → Gemini-2.5-Pro → [NaiveRAG || DuckDuckGo] → Unified JSON`
7. **Config 7**: `Prompt → Gemini-2.5-Pro → [HybridRAG || Tavily] → Unified JSON`
8. **Config 8**: `Prompt → Gemini-2.5-Pro → [HybridRAG || DuckDuckGo] → Unified JSON`

**Note**: `[A || B]` indicates parallel execution where both A and B run simultaneously, then merge results.

## Execution Workflow

**Each configuration executes parallel branches in a single workflow:**

```
1. Receive user prompt
2. Initialize LLM model (GPT-5 or Gemini 2.5 Pro)
3. PARALLEL EXECUTION (using LangGraph):
   
   RAG Branch:                        MCP Branch:
   ├─ Retrieve from Qdrant           ├─ Search with Tavily/DuckDuckGo
   ├─ Generate answer (RAG context)  ├─ Generate answer (MCP context)
   └─ RAGAS evaluation               └─ RAGAS evaluation

4. Merge results into unified state
5. Format single JSON output with both RAG and MCP results
6. Save unified JSON file
```

**Key Points**:
- Both branches execute **simultaneously** (parallel)
- Contexts remain **isolated** (never mixed)
- Results **merge** into single JSON for comparison
- Each branch gets independent RAGAS evaluation

## Main Execution Function

Create a unified workflow function with parallel execution:

```python
async def execute_unified_workflow(
    prompt: str,
    model: str,  # "gpt-5" | "gemini-2.5-pro"
    rag_type: str,  # "naive" | "hybrid"
    mcp_server: str  # "tavily" | "duckduckgo"
) -> dict:
    """
    Execute unified workflow with parallel RAG and MCP branches.
    Returns single JSON with both results.
    """
    # Using LangGraph for parallel execution
    # Both branches run simultaneously
    # Results merge into unified output
    pass
```

**Configuration list for iteration**:
```python
configurations = [
    {"model": "gpt-5", "rag_type": "naive", "mcp_server": "tavily"},
    {"model": "gpt-5", "rag_type": "naive", "mcp_server": "duckduckgo"},
    {"model": "gpt-5", "rag_type": "hybrid", "mcp_server": "tavily"},
    {"model": "gpt-5", "rag_type": "hybrid", "mcp_server": "duckduckgo"},
    {"model": "gemini-2.5-pro", "rag_type": "naive", "mcp_server": "tavily"},
    {"model": "gemini-2.5-pro", "rag_type": "naive", "mcp_server": "duckduckgo"},
    {"model": "gemini-2.5-pro", "rag_type": "hybrid", "mcp_server": "tavily"},
    {"model": "gemini-2.5-pro", "rag_type": "hybrid", "mcp_server": "duckduckgo"},
]

for config in configurations:
    # Single execution returns unified JSON
    unified_result = await execute_unified_workflow(prompt, **config)
    save_to_json(unified_result, f"{config['model']}_{config['rag_type']}_{config['mcp_server']}.json")
```

## Code Conventions

### Code Style
- Follow PEP 8 for Python
- Use type hints in all functions
- Document functions with Google-style docstrings

### Variable and Function Names
- Variables: `snake_case`
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

### Project Structure
```
project/
├── config/
│   ├── docker-compose.yml
│   └── mcp_config.json
├── src/
│   ├── rag/
│   │   ├── naive_rag.py
│   │   └── hybrid_rag.py
│   ├── mcp/
│   │   ├── mcp_client.py
│   │   └── tool_router.py
│   ├── evaluation/
│   │   └── ragas_evaluator.py
│   ├── models/
│   │   └── llm_factory.py
│   └── main.py
├── data/
│   ├── knowledge_base/
│   │   ├── chunks/
│   │   ├── embeddings/
│   │   └── originals/
│   └── outputs/
```

## Critical Documentation References

**For all implementations related to MCP, LangChain, and Qdrant**:

1. **Use the MCP server `Docs by LangChain` and `context7`** for access to updated documentation:
   - LangChain documentation
   - LangGraph documentation
   - Qdrant integration guides

2. **Check official documentation**:
   - LangChain MCP Adapters: https://github.com/langchain-ai/langchain-mcp-adapters
   - RAGAS: https://docs.ragas.io/
   - Qdrant + LangChain: https://qdrant.tech/documentation/frameworks/langchain/

## Important Notes

- **No simulated data**: All data must be real, coming from the configured sources
- **Docker is mandatory**: Qdrant, Tavily, and DuckDuckGo will be running in Docker containers
- **One MCP tool per execution**: The system must ensure that only Tavily or DuckDuckGo is used in each run, never both simultaneously
- **Reproducibility**: Each execution must be replicable with the same configuration