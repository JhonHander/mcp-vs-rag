# Quick Reference: Unified Workflow Architecture

## The Big Picture

```
ONE Configuration → ONE Execution → ONE JSON File (with BOTH results)
```

## What Happens in One Run

```
Config: gpt-4o + naive + tavily
         ↓
    Execute Once
         ↓
    ┌────┴────┐
    │         │
  RAG      MCP
  Path     Path
    │         │
    └────┬────┘
         ↓
   Unified JSON
   {
     rag_results: {...},
     mcp_results: {...}
   }
```

## Example Output

### File: `gpt4o_naive_tavily_1.json`

```json
{
  "configuration": {
    "model": "gpt-4o",
    "rag_type": "naive",
    "mcp_server": "tavily"
  },
  
  "rag_results": {
    "retrieved_context": ["From Qdrant..."],
    "generated_answer": "Based on static knowledge...",
    "ragas_metrics": {
      "answer_relevancy": 0.87,
      "faithfulness": 0.92
    }
  },
  
  "mcp_results": {
    "retrieved_context": ["From Tavily web search..."],
    "generated_answer": "Based on current web info...",
    "ragas_metrics": {
      "answer_relevancy": 0.82,
      "faithfulness": 0.89
    }
  }
}
```

## Key Facts

| Aspect | Details |
|--------|---------|
| **Executions per config** | 1 (not 2) |
| **Files per config** | 1 unified JSON |
| **Total files from 8 configs** | 8 unified JSONs |
| **Total evaluated results** | 16 (8 RAG + 8 MCP in 8 files) |
| **Execution model** | Parallel (RAG ‖ MCP) |
| **Context mixing** | Never (isolated) |
| **Output format** | Unified JSON with both |

## Why This Approach?

✅ **Parallel = Faster**: Both run at same time  
✅ **Isolated = Fair**: No context contamination  
✅ **Unified = Convenient**: One file, easy comparison  
✅ **Same Conditions**: Same prompt, model, execution time

## Code Entry Point

```python
from src.workflow.main_workflow import execute_unified_workflow

result = await execute_unified_workflow(
    prompt="Your question here",
    model_name="gpt-4o",
    rag_type="naive",
    mcp_server="tavily"
)

# result contains both rag_results and mcp_results
```

## Mental Model

Think of it as:
- **One workflow** with two parallel tracks (like railway tracks)
- Both trains (RAG and MCP) start together
- They never cross paths (isolated)
- Both arrive at the same station (unified JSON)
- You get both journey reports in one ticket (file)
