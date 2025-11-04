import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict
from src.workflow.main_workflow import execute_unified_workflow

# Experiment configurations
# Each config produces ONE unified JSON with both RAG and MCP results
CONFIGURATIONS = [
    {"model": "gpt-4o", "rag_type": "naive", "mcp_server": "tavily"},
    {"model": "gpt-4o", "rag_type": "naive", "mcp_server": "duckduckgo"},
    {"model": "gpt-4o", "rag_type": "hybrid", "mcp_server": "tavily"},
    {"model": "gpt-4o", "rag_type": "hybrid", "mcp_server": "duckduckgo"},
    {"model": "gemini-1.5-pro", "rag_type": "naive", "mcp_server": "tavily"},
    {"model": "gemini-1.5-pro", "rag_type": "naive", "mcp_server": "duckduckgo"},
    {"model": "gemini-1.5-pro", "rag_type": "hybrid", "mcp_server": "tavily"},
    {"model": "gemini-1.5-pro", "rag_type": "hybrid", "mcp_server": "duckduckgo"},
]


async def run_single_experiment(prompt: str, config: Dict[str, str]) -> Dict:
    """
    Run a single experiment configuration.
    Returns ONE unified JSON with both RAG and MCP results.
    """
    print(f"Running: {config}")

    try:
        # Execute unified workflow with parallel RAG and MCP branches
        result = await execute_unified_workflow(
            prompt=prompt,
            model_name=config["model"],
            rag_type=config["rag_type"],
            mcp_server=config["mcp_server"]
        )

        print(f"✅ Completed: {config}")
        return result

    except Exception as e:
        print(f"❌ Error in {config}: {str(e)}")

        # Return error result with empty RAG and MCP sections
        return {
            "execution_id": f"error_{datetime.now().isoformat()}",
            "timestamp": datetime.now().isoformat(),
            "configuration": config,
            "prompt": prompt,
            "error": str(e),
            "rag_results": {
                "retrieved_context": [],
                "generated_answer": f"Error: {str(e)}",
                "ragas_metrics": {"answer_relevancy": 0.0, "faithfulness": 0.0}
            },
            "mcp_results": {
                "retrieved_context": [],
                "generated_answer": f"Error: {str(e)}",
                "ragas_metrics": {"answer_relevancy": 0.0, "faithfulness": 0.0}
            }
        }


def save_result(result: Dict, filename: str = None, output_dir: str = "data/outputs"):
    """Save experiment result to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f"{result['execution_id']}.json"

    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"📁 Saved: {filename}")


async def run_full_experiment(prompt: str):
    """Run the complete experiment with all configurations."""
    print("🚀 Starting MCP vs RAG Comparison Experiment")
    print(f"📝 Prompt: {prompt}")
    print(f"🔧 Total configurations: {len(CONFIGURATIONS)}")
    print(f"📊 Each result contains BOTH RAG and MCP outputs")
    print("-" * 50)

    results = []

    for i, config in enumerate(CONFIGURATIONS, 1):
        print(f"\n[{i}/{len(CONFIGURATIONS)}]", end=" ")

        # Get unified result with both RAG and MCP
        result = await run_single_experiment(prompt, config)
        results.append(result)

        # Save unified result with descriptive filename
        filename = f"{config['model']}_{config['rag_type']}_{config['mcp_server']}_{i}.json"
        save_result(result, filename=filename)

        # Small delay between experiments
        await asyncio.sleep(1)

    # Save summary
    summary = {
        "experiment_summary": {
            "total_configurations": len(CONFIGURATIONS),
            "total_unified_results": len(results),
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "configurations": CONFIGURATIONS
        },
        "results": results
    }

    summary_path = os.path.join("data/outputs", "experiment_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Experiment completed!")
    print(f"� Total unified results: {len(results)}")
    print(f"💾 Summary saved to: {summary_path}")
    return results

if __name__ == "__main__":
    # Test prompt - you can modify this
    test_prompt = "What are the latest developments in artificial intelligence and machine learning in 2024?"

    # Run experiment
    asyncio.run(run_full_experiment(test_prompt))
