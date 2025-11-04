# MCP vs RAG Comparison Project

A research project comparing static retrieval (RAG) with dynamic web search (MCP) for question-answering systems.

## Project Structure

```
mcp-vs-rag/
├── config/                 # Configuration files
├── src/                   # Source code
│   ├── rag/              # RAG implementations  
│   ├── mcp/              # MCP client code
│   ├── models/           # LLM factory
│   ├── evaluation/       # RAGAS evaluator
│   └── workflow/         # Main workflow orchestration
├── data/                 # Data storage
│   ├── knowledge_base/   # RAG data (chunks, embeddings, originals)
│   └── outputs/          # Experiment results
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── run_experiment.py    # Main experiment runner
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment setup:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start Qdrant (if not already running):**
   ```bash
   cd config
   docker-compose up -d
   ```

## Usage

Run the complete experiment:
```bash
python run_experiment.py
```

This will execute 8 configurations comparing:
- Models: GPT-4o vs Gemini-1.5-Pro  
- RAG Types: Naive vs Hybrid
- MCP Servers: Tavily vs DuckDuckGo

Results are saved to `data/outputs/` as individual JSON files plus a summary.

## Configuration

Modify `run_experiment.py` to:
- Change the test prompt
- Adjust experiment configurations
- Customize output formats

## Requirements

- Python 3.8+
- Docker (for Qdrant)
- API keys for OpenAI, Google, Tavily
- Running MCP servers (Tavily, DuckDuckGo)