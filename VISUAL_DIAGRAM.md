# Visual Workflow Diagram

## Complete Execution Flow

```
╔══════════════════════════════════════════════════════════════════════╗
║                       USER INPUT & CONFIGURATION                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Question: "What are the latest AI developments?"                    ║
║  Model: gpt-4o                                                       ║
║  RAG Type: naive                                                     ║
║  MCP Server: tavily                                                  ║
║                                                                       ║
╚═══════════════════════════════╤══════════════════════════════════════╝
                                │
                                ▼
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                ┃   Initialize Workflow     ┃
                ┃   (UnifiedWorkflowState)  ┃
                ┗━━━━━━━━━━━┯━━━━━━━━━━━━━━━┛
                            │
                            ▼
        ╔═══════════════════════════════════════════════╗
        ║         PARALLEL EXECUTION NODE               ║
        ║        (asyncio.gather starts both)           ║
        ╚═══════════════════════════════════════════════╝
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┏━━━━━━━━━━━━━━━━━┓       ┏━━━━━━━━━━━━━━━━━┓
    ┃   RAG BRANCH    ┃       ┃   MCP BRANCH    ┃
    ┃   (Isolated)    ┃       ┃   (Isolated)    ┃
    ┗━━━━━━━━━━━━━━━━━┛       ┗━━━━━━━━━━━━━━━━━┛
              │                           │
              ▼                           ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  STEP 1         │       │  STEP 1         │
    │  Retrieve from  │       │  Search using   │
    │  Qdrant DB      │       │  Tavily API     │
    │                 │       │                 │
    │  Result:        │       │  Result:        │
    │  ["Context 1",  │       │  "Web results   │
    │   "Context 2"]  │       │   from Tavily"  │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  STEP 2         │       │  STEP 2         │
    │  Generate       │       │  Generate       │
    │  Answer with    │       │  Answer with    │
    │  GPT-4o using   │       │  GPT-4o using   │
    │  RAG context    │       │  MCP context    │
    │                 │       │                 │
    │  Result:        │       │  Result:        │
    │  "Based on      │       │  "According to  │
    │   documents..." │       │   web sources..."│
    └────────┬────────┘       └────────┬────────┘
             │                         │
             ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  STEP 3         │       │  STEP 3         │
    │  RAGAS          │       │  RAGAS          │
    │  Evaluation     │       │  Evaluation     │
    │                 │       │                 │
    │  Metrics:       │       │  Metrics:       │
    │  - Relevancy:   │       │  - Relevancy:   │
    │    0.87         │       │    0.82         │
    │  - Faithfulness:│       │  - Faithfulness:│
    │    0.92         │       │    0.89         │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             │    BOTH COMPLETE        │
             └─────────────┬───────────┘
                           │
                           ▼
              ┏━━━━━━━━━━━━━━━━━━━━━━┓
              ┃   MERGE RESULTS       ┃
              ┃   (in workflow state) ┃
              ┗━━━━━━━━━━━┯━━━━━━━━━━━┛
                          │
                          ▼
        ╔═════════════════════════════════════════╗
        ║      FORMAT UNIFIED JSON OUTPUT         ║
        ╠═════════════════════════════════════════╣
        ║ {                                       ║
        ║   "execution_id": "abc-123",            ║
        ║   "configuration": {                    ║
        ║     "model": "gpt-4o",                  ║
        ║     "rag_type": "naive",                ║
        ║     "mcp_server": "tavily"              ║
        ║   },                                    ║
        ║   "rag_results": {                      ║
        ║     "retrieved_context": [...],         ║
        ║     "generated_answer": "...",          ║
        ║     "ragas_metrics": {                  ║
        ║       "answer_relevancy": 0.87,         ║
        ║       "faithfulness": 0.92              ║
        ║     }                                   ║
        ║   },                                    ║
        ║   "mcp_results": {                      ║
        ║     "retrieved_context": [...],         ║
        ║     "generated_answer": "...",          ║
        ║     "ragas_metrics": {                  ║
        ║       "answer_relevancy": 0.82,         ║
        ║       "faithfulness": 0.89              ║
        ║     }                                   ║
        ║   }                                     ║
        ║ }                                       ║
        ╚═════════════════════════════════════════╝
                          │
                          ▼
              ┏━━━━━━━━━━━━━━━━━━━━━━┓
              ┃   SAVE TO FILE        ┃
              ┃   gpt4o_naive_        ┃
              ┃   tavily_1.json       ┃
              ┗━━━━━━━━━━━━━━━━━━━━━━┛
```

## Timing Diagram

```
Time →
├─────────────────────────────────────────────────┤
│                                                 │
│  [Start]                                        │
│    ↓                                            │
│  Initialize State (instant)                     │
│    ↓                                            │
│  ┌──────────────── PARALLEL ────────────────┐  │
│  │                                           │  │
│  │  RAG Branch:                              │  │
│  │  ├─ Retrieve (2s)                         │  │
│  │  ├─ Generate (3s)                         │  │
│  │  └─ Evaluate (1s)                         │  │
│  │  Total: 6s                                │  │
│  │                                           │  │
│  │  MCP Branch:                              │  │
│  │  ├─ Search (3s)                           │  │
│  │  ├─ Generate (3s)                         │  │
│  │  └─ Evaluate (1s)                         │  │
│  │  Total: 7s                                │  │
│  │                                           │  │
│  └─────────────── Wait for Both ────────────┘  │
│    ↓                                            │
│  Actual Total Time: ~7s (not 13s!)             │
│    ↓                                            │
│  Merge & Format (instant)                       │
│    ↓                                            │
│  [End]                                          │
│                                                 │
└─────────────────────────────────────────────────┘

Performance Gain: ~46% faster than sequential
```

## Context Flow

```
┌──────────────────────────────────────────────────────────┐
│                    USER QUESTION                          │
│         "What are the latest AI developments?"            │
└─────────────────────────┬────────────────────────────────┘
                          │
                          │ (shared input)
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
┌─────────────────┐              ┌─────────────────┐
│  RAG CONTEXT    │              │  MCP CONTEXT    │
│  (ISOLATED)     │              │  (ISOLATED)     │
│                 │              │                 │
│  From Qdrant:   │              │  From Tavily:   │
│  • Doc about ML │              │  • News article │
│  • Doc about AI │              │  • Blog post    │
│  • Research     │              │  • Recent paper │
│                 │              │                 │
│  ✗ NO MCP DATA  │              │  ✗ NO RAG DATA  │
└────────┬────────┘              └────────┬────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────┐              ┌─────────────────┐
│  LLM sees ONLY  │              │  LLM sees ONLY  │
│  RAG context    │              │  MCP context    │
└────────┬────────┘              └────────┬────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────┐              ┌─────────────────┐
│  RAG Answer     │              │  MCP Answer     │
│  (from static   │              │  (from web      │
│   knowledge)    │              │   search)       │
└────────┬────────┘              └────────┬────────┘
         │                                 │
         └────────────────┬────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  BOTH ANSWERS IN      │
              │  SINGLE JSON          │
              │  (for comparison)     │
              └───────────────────────┘
```

## File Output Structure

```
data/outputs/
├── gpt4o_naive_tavily_1.json          ← Config 1
│   ├── rag_results (from Qdrant)
│   └── mcp_results (from Tavily)
│
├── gpt4o_naive_duckduckgo_2.json      ← Config 2
│   ├── rag_results (from Qdrant)
│   └── mcp_results (from DuckDuckGo)
│
├── gpt4o_hybrid_tavily_3.json         ← Config 3
│   ├── rag_results (from Qdrant)
│   └── mcp_results (from Tavily)
│
... (8 files total, one per configuration)
│
└── experiment_summary.json             ← All results
    └── results: [all 8 unified JSONs]
```

## Legend

```
┏━━━━━━━┓  Major process/component
┃       ┃
┗━━━━━━━┛

╔═══════╗  Important section/grouping
║       ║
╚═══════╝

┌───────┐  Sub-process/step
│       │
└───────┘

│  Flow connection (sequential)
▼

├──┤  Parallel split/join

[A || B]  Parallel execution of A and B
```
