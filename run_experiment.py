import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict
from src.workflow.main_workflow import execute_unified_workflow
from data.ground_truth import load_ground_truth_dataset

# Experiment configurations
# Each config produces ONE unified JSON with both RAG and MCP results
CONFIGURATIONS = [
    {"model": "gpt-5", "rag_type": "naive", "mcp_server": "tavily"},
    {"model": "gpt-5", "rag_type": "naive", "mcp_server": "duckduckgo"},
    {"model": "gpt-5", "rag_type": "hybrid", "mcp_server": "tavily"},
    {"model": "gpt-5", "rag_type": "hybrid", "mcp_server": "duckduckgo"},
    {"model": "gemini-2.5-pro", "rag_type": "naive", "mcp_server": "tavily"},
    {"model": "gemini-2.5-pro", "rag_type": "naive", "mcp_server": "duckduckgo"},
    {"model": "gemini-2.5-pro", "rag_type": "hybrid", "mcp_server": "tavily"},
    {"model": "gemini-2.5-pro", "rag_type": "hybrid", "mcp_server": "duckduckgo"},
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

        print(f"[SUCCESS] Completed: {config}")
        return result

    except Exception as e:
        print(f"[ERROR] Failed in {config}: {str(e)}")

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

    print(f"[SAVED] {filename}")


def calculate_consolidated_analysis(results: List[Dict], prompt: str) -> Dict:
    """
    Calculate consolidated analysis with comparative metrics.
    Groups results by model, RAG type, and MCP server with averages.
    """
    analysis = {
        "metadata": {
            "experiment_date": datetime.now().isoformat(),
            "total_configurations": len(results),
            "prompt": prompt
        },
        "comparative_analysis": {
            "by_model": {},
            "by_rag_type": {},
            "by_mcp_server": {}
        },
        "detailed_results": []
    }

    # Collect metrics for each configuration
    for idx, result in enumerate(results, 1):
        config = result.get("configuration", {})
        model = config.get("model", "unknown")
        rag_type = config.get("rag_type", "unknown")
        mcp_server = config.get("mcp_server", "unknown")

        rag_metrics = result.get("rag_results", {}).get("ragas_metrics", {})
        mcp_metrics = result.get("mcp_results", {}).get("ragas_metrics", {})

        # Initialize model grouping
        if model not in analysis["comparative_analysis"]["by_model"]:
            analysis["comparative_analysis"]["by_model"][model] = {
                "rag_relevancy_scores": [],
                "rag_faithfulness_scores": [],
                "mcp_relevancy_scores": [],
                "mcp_faithfulness_scores": [],
                "configurations_tested": 0
            }

        # Initialize RAG type grouping
        if rag_type not in analysis["comparative_analysis"]["by_rag_type"]:
            analysis["comparative_analysis"]["by_rag_type"][rag_type] = {
                "relevancy_scores": [],
                "faithfulness_scores": [],
                "configurations_tested": 0
            }

        # Initialize MCP server grouping
        if mcp_server not in analysis["comparative_analysis"]["by_mcp_server"]:
            analysis["comparative_analysis"]["by_mcp_server"][mcp_server] = {
                "relevancy_scores": [],
                "faithfulness_scores": [],
                "configurations_tested": 0
            }

        # Add scores to model grouping
        analysis["comparative_analysis"]["by_model"][model]["rag_relevancy_scores"].append(
            rag_metrics.get("answer_relevancy", 0.0))
        analysis["comparative_analysis"]["by_model"][model]["rag_faithfulness_scores"].append(
            rag_metrics.get("faithfulness", 0.0))
        analysis["comparative_analysis"]["by_model"][model]["mcp_relevancy_scores"].append(
            mcp_metrics.get("answer_relevancy", 0.0))
        analysis["comparative_analysis"]["by_model"][model]["mcp_faithfulness_scores"].append(
            mcp_metrics.get("faithfulness", 0.0))
        analysis["comparative_analysis"]["by_model"][model]["configurations_tested"] += 1

        # Add scores to RAG type grouping
        analysis["comparative_analysis"]["by_rag_type"][rag_type]["relevancy_scores"].append(
            rag_metrics.get("answer_relevancy", 0.0))
        analysis["comparative_analysis"]["by_rag_type"][rag_type]["faithfulness_scores"].append(
            rag_metrics.get("faithfulness", 0.0))
        analysis["comparative_analysis"]["by_rag_type"][rag_type]["configurations_tested"] += 1

        # Add scores to MCP server grouping
        analysis["comparative_analysis"]["by_mcp_server"][mcp_server]["relevancy_scores"].append(
            mcp_metrics.get("answer_relevancy", 0.0))
        analysis["comparative_analysis"]["by_mcp_server"][mcp_server]["faithfulness_scores"].append(
            mcp_metrics.get("faithfulness", 0.0))
        analysis["comparative_analysis"]["by_mcp_server"][mcp_server]["configurations_tested"] += 1

        # Add to detailed results with file reference
        analysis["detailed_results"].append({
            "config_id": idx,
            "model": model,
            "rag_type": rag_type,
            "mcp_server": mcp_server,
            "rag_metrics": rag_metrics,
            "mcp_metrics": mcp_metrics,
            "file_reference": f"{model}_{rag_type}_{mcp_server}_{idx}.json"
        })

    # Calculate averages for model grouping
    for model_data in analysis["comparative_analysis"]["by_model"].values():
        model_data["average_rag_relevancy"] = sum(
            model_data["rag_relevancy_scores"]) / len(model_data["rag_relevancy_scores"]) if model_data["rag_relevancy_scores"] else 0.0
        model_data["average_rag_faithfulness"] = sum(
            model_data["rag_faithfulness_scores"]) / len(model_data["rag_faithfulness_scores"]) if model_data["rag_faithfulness_scores"] else 0.0
        model_data["average_mcp_relevancy"] = sum(
            model_data["mcp_relevancy_scores"]) / len(model_data["mcp_relevancy_scores"]) if model_data["mcp_relevancy_scores"] else 0.0
        model_data["average_mcp_faithfulness"] = sum(
            model_data["mcp_faithfulness_scores"]) / len(model_data["mcp_faithfulness_scores"]) if model_data["mcp_faithfulness_scores"] else 0.0
        # Remove individual scores, keep only averages
        del model_data["rag_relevancy_scores"]
        del model_data["rag_faithfulness_scores"]
        del model_data["mcp_relevancy_scores"]
        del model_data["mcp_faithfulness_scores"]

    # Calculate averages for RAG type grouping
    for rag_data in analysis["comparative_analysis"]["by_rag_type"].values():
        rag_data["average_relevancy"] = sum(
            rag_data["relevancy_scores"]) / len(rag_data["relevancy_scores"]) if rag_data["relevancy_scores"] else 0.0
        rag_data["average_faithfulness"] = sum(
            rag_data["faithfulness_scores"]) / len(rag_data["faithfulness_scores"]) if rag_data["faithfulness_scores"] else 0.0
        del rag_data["relevancy_scores"]
        del rag_data["faithfulness_scores"]

    # Calculate averages for MCP server grouping
    for mcp_data in analysis["comparative_analysis"]["by_mcp_server"].values():
        mcp_data["average_relevancy"] = sum(
            mcp_data["relevancy_scores"]) / len(mcp_data["relevancy_scores"]) if mcp_data["relevancy_scores"] else 0.0
        mcp_data["average_faithfulness"] = sum(
            mcp_data["faithfulness_scores"]) / len(mcp_data["faithfulness_scores"]) if mcp_data["faithfulness_scores"] else 0.0
        del mcp_data["relevancy_scores"]
        del mcp_data["faithfulness_scores"]

    return analysis


async def run_full_experiment(prompt: str):
    """Run the complete experiment with all configurations."""
    print("="*60)
    print("MCP vs RAG Comparison Experiment")
    print("="*60)
    print(f"Prompt: {prompt}")
    print(f"Total configurations: {len(CONFIGURATIONS)}")
    print(f"Each result contains BOTH RAG and MCP outputs")
    print("-" * 60)

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

    # Save full summary with all results
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

    # Save consolidated analysis with comparative metrics
    consolidated = calculate_consolidated_analysis(results, prompt)
    consolidated_path = os.path.join(
        "data/outputs", "consolidated_analysis.json")
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("Experiment completed!")
    print(f"Total unified results: {len(results)}")
    print(f"Full summary saved to: {summary_path}")
    print(f"Consolidated analysis saved to: {consolidated_path}")
    print("="*60)
    return results


async def run_experiment_with_dataset(use_all_questions: bool = False, num_questions: int = 1):
    """
    Run experiment using questions from ground truth dataset.

    Args:
        use_all_questions: If True, run experiment with all questions in dataset
        num_questions: If use_all_questions=False, run with this many random questions
    """
    try:
        # Load ground truth dataset
        print("Loading ground truth dataset...")
        dataset = load_ground_truth_dataset()
        dataset.print_summary()

        # Get questions to test
        if use_all_questions:
            questions = dataset.get_all_questions()
            print(f"\nRunning experiment with ALL {len(questions)} questions")
        else:
            questions = dataset.get_random_questions(num_questions)
            print(
                f"\nRunning experiment with {len(questions)} random question(s)")

        # Run experiments for each question
        all_results = []

        for q_idx, question_data in enumerate(questions, 1):
            question_id = question_data['id']
            question_text = question_data['question']
            ground_truth = question_data['ground_truth']

            print(f"\n{'='*60}")
            print(f"Question {q_idx}/{len(questions)} (ID: {question_id})")
            print(f"Q: {question_text[:100]}...")
            print(f"{'='*60}")

            # Run experiment with this question
            results = await run_full_experiment(question_text)

            # Add ground truth to results
            for result in results:
                result['ground_truth'] = ground_truth
                result['question_id'] = question_id

            all_results.extend(results)

            # Save per-question summary
            question_summary = {
                "question_id": question_id,
                "question": question_text,
                "ground_truth": ground_truth,
                "results": results
            }

            filename = f"question_{question_id}_results.json"
            save_result(question_summary, filename=filename)

        print(f"\n{'='*60}")
        print(f"Completed experiment with {len(questions)} question(s)")
        print(f"Total results: {len(all_results)}")
        print(f"{'='*60}")

        return all_results

    except FileNotFoundError:
        print("\n[ERROR] Ground truth dataset not found!")
        print("Please follow these steps:")
        print("1. Copy your Excel file to: data/ground_truth/preguntas.xlsx")
        print("2. Run: python data/ground_truth/process_dataset.py")
        print("3. Run this experiment again")
        return None
    except Exception as e:
        print(f"\n[ERROR] Failed to run experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Run with all questions
            asyncio.run(run_experiment_with_dataset(use_all_questions=True))
        elif sys.argv[1].isdigit():
            # Run with N random questions
            num = int(sys.argv[1])
            asyncio.run(run_experiment_with_dataset(
                use_all_questions=False, num_questions=num))
        else:
            print("Usage:")
            print("  python run_experiment.py           # Run with 1 random question")
            print("  python run_experiment.py 5         # Run with 5 random questions")
            print("  python run_experiment.py --all     # Run with all questions")
    else:
        # Default: run with 1 random question
        asyncio.run(run_experiment_with_dataset(
            use_all_questions=False, num_questions=1))
