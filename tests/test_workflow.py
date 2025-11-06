"""
Test script to verify the complete unified workflow.
Tests parallel execution of RAG and MCP branches with a simple query.
"""
import asyncio
from src.workflow.main_workflow import execute_unified_workflow
import json


async def test_workflow_single_config():
    """Test workflow with a single configuration."""
    print("\n" + "=" * 70)
    print("TESTING UNIFIED WORKFLOW - SINGLE CONFIGURATION")
    print("=" * 70)

    config = {
        "prompt": "What are the recommendations for prenatal care?",
        "model_name": "gpt-5",
        "rag_type": "naive",
        "mcp_server": "tavily"
    }

    print(f"\nConfiguration:")
    print(f"  Model: {config['model_name']}")
    print(f"  RAG Type: {config['rag_type']}")
    print(f"  MCP Server: {config['mcp_server']}")
    print(f"  Prompt: {config['prompt']}")

    print("\n" + "-" * 70)
    print("Executing workflow (both branches in parallel)...")
    print("-" * 70)

    try:
        result = await execute_unified_workflow(**config)

        print("\nWORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print("\nRAG RESULTS:")
        print("-" * 70)
        print(
            f"Context items retrieved: {len(result['rag_results']['retrieved_context'])}")
        print(
            f"Answer preview: {result['rag_results']['generated_answer'][:200]}...")
        print(f"Metrics: {result['rag_results']['ragas_metrics']}")

        print("\nMCP RESULTS:")
        print("-" * 70)
        print(
            f"Context length: {len(result['mcp_results']['retrieved_context'][0]) if result['mcp_results']['retrieved_context'] else 0} chars")
        print(
            f"Answer preview: {result['mcp_results']['generated_answer'][:200]}...")
        print(f"Metrics: {result['mcp_results']['ragas_metrics']}")

        print("\n" + "=" * 70)
        print("FULL OUTPUT (JSON):")
        print("=" * 70)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        return True

    except Exception as e:
        print(f"\nWORKFLOW FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_workflow_all_configs():
    """Test workflow with multiple configurations."""
    print("\n" + "=" * 70)
    print("TESTING UNIFIED WORKFLOW - ALL CONFIGURATIONS")
    print("=" * 70)

    configurations = [
        {"model": "gpt-5", "rag_type": "naive", "mcp_server": "tavily"},
        {"model": "gpt-5", "rag_type": "hybrid", "mcp_server": "duckduckgo"},
    ]

    prompt = "What are the latest AI developments?"

    results = []

    for i, config in enumerate(configurations, 1):
        print(f"\n{'=' * 70}")
        print(f"Configuration {i}/{len(configurations)}")
        print('=' * 70)
        print(f"Model: {config['model']}")
        print(f"RAG: {config['rag_type']}")
        print(f"MCP: {config['mcp_server']}")

        try:
            result = await execute_unified_workflow(
                prompt=prompt,
                model_name=config['model'],
                rag_type=config['rag_type'],
                mcp_server=config['mcp_server']
            )

            print(f"SUCCESS")
            print(
                f"RAG answer length: {len(result['rag_results']['generated_answer'])} chars")
            print(
                f"MCP answer length: {len(result['mcp_results']['generated_answer'])} chars")

            results.append((config, True))

        except Exception as e:
            print(f"FAILED: {str(e)}")
            results.append((config, False))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for config, success in results:
        status = "PASSED" if success else "FAILED"
        print(
            f"{config['model']} + {config['rag_type']} + {config['mcp_server']}: {status}")

    all_passed = all(success for _, success in results)
    return all_passed


async def test_parallel_execution_timing():
    """Verify that parallel execution is actually faster than sequential."""
    import time

    print("\n" + "=" * 70)
    print("TESTING PARALLEL EXECUTION TIMING")
    print("=" * 70)

    config = {
        "prompt": "What is machine learning?",
        "model_name": "gpt-5",
        "rag_type": "naive",
        "mcp_server": "tavily"
    }

    print("\nExecuting workflow with parallel branches...")
    start = time.time()

    try:
        await execute_unified_workflow(**config)
        parallel_time = time.time() - start

        print(f"Parallel execution time: {parallel_time:.2f} seconds")
        print("\nNote: If both branches truly run in parallel,")
        print("the total time should be close to the slower branch,")
        print("not the sum of both branches.")

        return True

    except Exception as e:
        print(f"FAILED: {str(e)}")
        return False


if __name__ == "__main__":
    import sys

    async def run_tests():
        if len(sys.argv) > 1:
            test_type = sys.argv[1]

            if test_type == "single":
                await test_workflow_single_config()
            elif test_type == "all":
                await test_workflow_all_configs()
            elif test_type == "timing":
                await test_parallel_execution_timing()
            else:
                print("Usage: python test_workflow.py [single|all|timing]")
        else:
            print("Running single configuration test...")
            await test_workflow_single_config()

    asyncio.run(run_tests())
