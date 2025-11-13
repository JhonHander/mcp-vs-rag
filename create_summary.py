"""
Script to consolidate all question results into a single summary file.
Combines all 10 question JSONs into one comprehensive summary.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_question_result(question_id: int) -> Dict[str, Any]:
    """Load results for a specific question."""
    file_path = f"data/outputs/question_{question_id}/question_{question_id}_results.json"
    
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: {file_path} not found")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_consolidated_summary(output_file: str = "data/outputs/consolidated_summary.json"):
    """
    Consolidate all 10 question results into a single summary file.
    
    Args:
        output_file: Path where the consolidated summary will be saved
    """
    print("=" * 80)
    print("Creating Consolidated Summary")
    print("=" * 80)
    
    all_questions = []
    total_executions = 0
    
    # Load all 10 questions
    for question_id in range(1, 11):
        print(f"\n📄 Loading question {question_id}...")
        result = load_question_result(question_id)
        
        if result:
            all_questions.append(result)
            total_executions += len(result.get('results', []))
            print(f"   ✓ Loaded {len(result.get('results', []))} executions")
        else:
            print(f"   ✗ Failed to load question {question_id}")
    
    # Create consolidated structure
    consolidated = {
        "summary": {
            "title": "Consolidated RAG vs MCP Experiment Results",
            "description": "Complete results from all 10 questions with multiple configurations",
            "total_questions": len(all_questions),
            "total_executions": total_executions,
            "timestamp": datetime.now().isoformat(),
            "configurations": {
                "models": ["gpt-5", "gemini-2.5-pro"],
                "rag_types": ["naive", "hybrid"],
                "mcp_servers": ["tavily", "duckduckgo"],
                "total_combinations": 8
            }
        },
        "questions": all_questions
    }
    
    # Calculate aggregate statistics
    print("\n" + "=" * 80)
    print("Calculating Statistics")
    print("=" * 80)
    
    total_rag_cost = 0.0
    total_mcp_cost = 0.0
    rag_metrics_sum = {"answer_relevancy": 0, "faithfulness": 0, "count": 0}
    mcp_metrics_sum = {"answer_relevancy": 0, "faithfulness": 0, "count": 0}
    
    for question in all_questions:
        for result in question.get('results', []):
            # Sum costs
            cost_summary = result.get('cost_summary', {})
            total_rag_cost += cost_summary.get('rag_cost_usd', 0)
            total_mcp_cost += cost_summary.get('mcp_cost_usd', 0)
            
            # Sum RAG metrics
            rag_metrics = result.get('rag_results', {}).get('ragas_metrics', {})
            if 'answer_relevancy' in rag_metrics:
                rag_metrics_sum['answer_relevancy'] += rag_metrics['answer_relevancy']
                rag_metrics_sum['faithfulness'] += rag_metrics.get('faithfulness', 0)
                rag_metrics_sum['count'] += 1
            
            # Sum MCP metrics
            mcp_metrics = result.get('mcp_results', {}).get('ragas_metrics', {})
            if 'answer_relevancy' in mcp_metrics:
                mcp_metrics_sum['answer_relevancy'] += mcp_metrics['answer_relevancy']
                mcp_metrics_sum['faithfulness'] += mcp_metrics.get('faithfulness', 0)
                mcp_metrics_sum['count'] += 1
    
    # Calculate averages
    avg_rag_relevancy = rag_metrics_sum['answer_relevancy'] / rag_metrics_sum['count'] if rag_metrics_sum['count'] > 0 else 0
    avg_rag_faithfulness = rag_metrics_sum['faithfulness'] / rag_metrics_sum['count'] if rag_metrics_sum['count'] > 0 else 0
    avg_mcp_relevancy = mcp_metrics_sum['answer_relevancy'] / mcp_metrics_sum['count'] if mcp_metrics_sum['count'] > 0 else 0
    avg_mcp_faithfulness = mcp_metrics_sum['faithfulness'] / mcp_metrics_sum['count'] if mcp_metrics_sum['count'] > 0 else 0
    
    consolidated["aggregate_statistics"] = {
        "costs": {
            "total_rag_cost_usd": round(total_rag_cost, 6),
            "total_mcp_cost_usd": round(total_mcp_cost, 6),
            "total_cost_usd": round(total_rag_cost + total_mcp_cost, 6),
            "average_rag_cost_per_execution": round(total_rag_cost / total_executions, 6) if total_executions > 0 else 0,
            "average_mcp_cost_per_execution": round(total_mcp_cost / total_executions, 6) if total_executions > 0 else 0
        },
        "rag_performance": {
            "average_answer_relevancy": round(avg_rag_relevancy, 4),
            "average_faithfulness": round(avg_rag_faithfulness, 4),
            "total_evaluations": rag_metrics_sum['count']
        },
        "mcp_performance": {
            "average_answer_relevancy": round(avg_mcp_relevancy, 4),
            "average_faithfulness": round(avg_mcp_faithfulness, 4),
            "total_evaluations": mcp_metrics_sum['count']
        }
    }
    
    # Save consolidated file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Consolidated summary saved to: {output_file}")
    print(f"\n📊 Summary Statistics:")
    print(f"   - Total Questions: {len(all_questions)}")
    print(f"   - Total Executions: {total_executions}")
    print(f"   - Total Cost: ${total_rag_cost + total_mcp_cost:.6f}")
    print(f"   - RAG Avg Relevancy: {avg_rag_relevancy:.4f}")
    print(f"   - MCP Avg Relevancy: {avg_mcp_relevancy:.4f}")
    print(f"   - RAG Avg Faithfulness: {avg_rag_faithfulness:.4f}")
    print(f"   - MCP Avg Faithfulness: {avg_mcp_faithfulness:.4f}")
    print("=" * 80)
    
    return consolidated


if __name__ == "__main__":
    create_consolidated_summary()
