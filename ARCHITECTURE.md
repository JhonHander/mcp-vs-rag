# Parallel Execution with Unified JSON Architecture

## Overview

This project implements a **unified LangGraph workflow** with **parallel RAG and MCP branches** that execute simultaneously and merge their results into a **single JSON output** for direct comparison.

## Architecture Diagram

```
                    User Prompt
                         │
                         ▼
                 Initialize State
                   (Model, RAG, MCP)
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
    ┌─────────────┐              ┌──────────────┐
    │ RAG Branch  │              │  MCP Branch  │
    │             │              │              │
    │ Step 1:     │              │  Step 1:     │
    │ Retrieve    │              │  Search Web  │
    │ from Qdrant │              │  (Tavily/    │
    │             │              │  DuckDuckGo) │
    │             │              │              │
    │ Step 2:     │              │  Step 2:     │
    │ Generate    │              │  Generate    │
    │ Answer      │              │  Answer      │
    │ (RAG ctx)   │              │  (MCP ctx)   │
    │             │              │              │
    │ Step 3:     │              │  Step 3:     │
    │ RAGAS Eval  │              │  RAGAS Eval  │
    └──────┬──────┘              └──────┬───────┘
           │                             │
           │    PARALLEL EXECUTION       │
           │    (asyncio.gather)         │
           │                             │
           └──────────────┬──────────────┘
                          ▼
                  Merge Results
                          │
                          ▼
              ┌───────────────────────┐
              │  Single Unified JSON  │
              │                       │
              │  ✓ Configuration      │
              │  ✓ RAG Results        │
              │  ✓ MCP Results        │
              │  ✓ Both RAGAS Metrics │
              └───────────────────────┘
```

## Key Principles

### 1. Parallel Execution
- **RAG and MCP branches run simultaneously** using `asyncio.gather()`
- Reduces total execution time (no sequential waiting)
- Both branches are independent and don't block each other

### 2. Context Isolation
- **RAG context** comes only from Qdrant (static knowledge base)
- **MCP context** comes only from web search (dynamic information)
- Contexts are **never mixed** - each LLM sees only one source
- Maintains purity of each approach for fair comparison

### 3. Unified Output
- **Single JSON file** per configuration
- Contains **both** RAG and MCP results side-by-side
- Easy comparison without managing multiple files
- Clear structure: `rag_results` and `mcp_results` sections

### 4. Independent Evaluation
- Each branch gets its **own RAGAS evaluation**
- Metrics calculated separately for fair assessment
- Can compare which approach performs better on same question

## LangGraph Implementation

### Workflow Structure

```python
class UnifiedWorkflow:
    def create_workflow(self):
        # Define parallel execution node
        async def execute_parallel_branches(state):
            # Run both branches simultaneously
            rag_result, mcp_result = await asyncio.gather(
                run_rag_branch(state),
                run_mcp_branch(state)
            )
            return merged_state
        
        # Build graph
        workflow = StateGraph(UnifiedWorkflowState)
        workflow.add_node("parallel_execution", execute_parallel_branches)
        workflow.add_edge(START, "parallel_execution")
        workflow.add_edge("parallel_execution", END)
```

### State Management

```python
class UnifiedWorkflowState(BaseModel):
    # Common fields
    execution_id: str
    prompt: str
    model_name: str
    rag_type: str
    mcp_server: str
    
    # RAG branch results
    rag_context: List[str]
    rag_answer: str
    rag_metrics: Dict[str, float]
    
    # MCP branch results
    mcp_context: str
    mcp_answer: str
    mcp_metrics: Dict[str, float]
```

## Output Structure

### Unified JSON Format

```json
{
  "execution_id": "uuid-here",
  "timestamp": "2025-11-03T10:30:00",
  "configuration": {
    "model": "gpt-4o",
    "rag_type": "naive",
    "mcp_server": "tavily"
  },
  "prompt": "What are the latest AI developments?",
  
  "rag_results": {
    "retrieved_context": [
      "Context from document 1...",
      "Context from document 2..."
    ],
    "generated_answer": "Based on the knowledge base...",
    "ragas_metrics": {
      "answer_relevancy": 0.87,
      "faithfulness": 0.92
    }
  },
  
  "mcp_results": {
    "retrieved_context": [
      "Web search results from Tavily..."
    ],
    "generated_answer": "Based on recent web sources...",
    "ragas_metrics": {
      "answer_relevancy": 0.82,
      "faithfulness": 0.89
    }
  }
}
```

### File Naming Convention

Files are named to reflect their complete configuration:
- `gpt4o_naive_tavily_1.json`
- `gpt4o_naive_duckduckgo_2.json`
- `gpt4o_hybrid_tavily_3.json`
- `gemini15pro_naive_tavily_5.json`

## Execution Flow

### Single Configuration Run

1. **Initialize**: Create `UnifiedWorkflowState` with config parameters
2. **Execute**: Run `execute_unified_workflow()`
   - Spawns parallel tasks for RAG and MCP
   - Both branches execute simultaneously
   - Wait for both to complete
3. **Merge**: Combine results into single state object
4. **Format**: Structure output as unified JSON
5. **Save**: Write to file with descriptive name

### Full Experiment (8 Configurations)

```python
for config in CONFIGURATIONS:
    result = await execute_unified_workflow(
        prompt=prompt,
        model_name=config["model"],
        rag_type=config["rag_type"],
        mcp_server=config["mcp_server"]
    )
    save_result(result, f"{config['model']}_{config['rag_type']}_{config['mcp_server']}.json")
```

## Total Results

From 8 configurations, you get:
- **8 unified JSON files** (one per configuration)
- **16 evaluated results** (8 RAG + 8 MCP, stored in 8 files)

Each file contains:
- 1 RAG evaluation (with metrics)
- 1 MCP evaluation (with metrics)

## Benefits of This Architecture

### 1. **Speed**
- Parallel execution cuts runtime nearly in half
- No waiting for sequential operations

### 2. **Clarity**
- Single file contains complete comparison
- No need to match separate RAG/MCP files
- Configuration clearly shown at top

### 3. **Fair Comparison**
- Same prompt, same model, same execution time
- Contexts remain isolated for purity
- Side-by-side metrics enable direct analysis

### 4. **Scalability**
- Easy to add more branches (e.g., hybrid RAG+MCP)
- LangGraph makes conditional routing simple
- State management handles complex workflows

### 5. **Maintainability**
- Clear separation of concerns
- Each branch is self-contained
- Easy to modify one without affecting others

## Code Structure

```
src/workflow/
└── main_workflow.py
    ├── UnifiedWorkflowState (state model)
    ├── UnifiedWorkflow (orchestrator)
    │   ├── retrieve_rag_context()
    │   ├── generate_rag_answer()
    │   ├── evaluate_rag()
    │   ├── search_mcp_context()
    │   ├── generate_mcp_answer()
    │   ├── evaluate_mcp()
    │   └── execute_parallel_branches()
    └── execute_unified_workflow() (main entry point)

run_experiment.py
├── CONFIGURATIONS (8 combinations)
├── run_single_experiment() (calls unified workflow)
└── run_full_experiment() (orchestrates all configs)
```

## Comparison Analysis

After running all experiments, you can analyze:

1. **RAG vs MCP Performance**
   - Which approach has better relevancy scores?
   - Which has better faithfulness?
   - Does it vary by model or RAG type?

2. **Model Comparison**
   - Does GPT-4o or Gemini-1.5-Pro perform better?
   - Are differences consistent across RAG/MCP?

3. **RAG Strategy**
   - Does Hybrid RAG outperform Naive RAG?
   - Is the improvement worth the complexity?

4. **MCP Tool Selection**
   - Does Tavily or DuckDuckGo provide better context?
   - Which integrates better with each model?

All comparisons are fair because both branches run under identical conditions within the same execution!
