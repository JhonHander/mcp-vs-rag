"""
Main workflow orchestration using LangGraph.
Implements parallel RAG and MCP branches that merge into a single unified JSON output.
Both pipelines execute independently using LangGraph's native parallelism.
"""

from datetime import datetime
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, START, END
import uuid

from src.rag.naive_rag import NaiveRAG
from src.rag.hybrid_rag import HybridRAG
from src.mcp.mcp_client import get_search_tool_for_server
from src.models.llm_factory import create_llm
from src.evaluation.ragas_evaluator import RAGASEvaluator
from src.config.mcp_config import get_mcp_search_config, should_truncate_context


class UnifiedWorkflowState(TypedDict):
    """
    Unified state for parallel RAG and MCP execution.
    Both branches populate their own fields independently.
    Fields that could be updated by parallel nodes use separate keys to avoid conflicts.
    """
    execution_id: str
    prompt: str
    model_name: str
    rag_type: str
    mcp_server: str
    timestamp: str

    rag_context: List[str]
    rag_answer: str
    rag_metrics: Dict[str, float]
    rag_generation_cost: Dict[str, Any]  # Cost tracking for RAG generation

    mcp_context: str
    mcp_answer: str
    mcp_metrics: Dict[str, float]
    mcp_generation_cost: Dict[str, Any]  # Cost tracking for MCP generation


class UnifiedWorkflow:
    """
    Unified workflow with parallel RAG and MCP branches using LangGraph's native parallelism.

    Architecture:
        START fans out to both branches:
        - RAG branch: retrieve_rag -> generate_rag -> evaluate_rag
        - MCP branch: search_mcp -> generate_mcp -> evaluate_mcp
        Both branches execute concurrently and converge at END.
    """

    def __init__(self):
        self.naive_rag = NaiveRAG()
        self.hybrid_rag = HybridRAG()
        self.evaluator = RAGASEvaluator()

    def create_workflow(self) -> StateGraph:
        """Create the unified workflow with LangGraph's native parallel execution."""

        def retrieve_rag_context(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """RAG Branch Step 1: Retrieve context from RAG."""
            if state["rag_type"] == "naive":
                contexts = self.naive_rag.retrieve(state["prompt"])
            else:
                contexts = self.hybrid_rag.retrieve(state["prompt"])
            return {"rag_context": contexts}

        def generate_rag_answer(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """RAG Branch Step 2: Generate answer using RAG context."""
            llm = create_llm(state["model_name"])
            rag_text = "\n".join(state["rag_context"])
            prompt_template = f"""Based on the following context, answer the question: {state["prompt"]}

Context (from knowledge base):
{rag_text}

Please provide a comprehensive answer based on the available information."""
            
            # Capture response and cost information
            response, cost = llm.invoke(prompt_template)
            
            return {
                "rag_answer": response.content,
                "rag_generation_cost": cost
            }

        def evaluate_rag(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """RAG Branch Step 3: Evaluate RAG answer with RAGAS."""
            metrics = self.evaluator.evaluate_response(
                question=state["prompt"],
                answer=state["rag_answer"],
                contexts=state["rag_context"]
            )
            return {"rag_metrics": metrics}

        async def search_mcp_context(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """MCP Branch Step 1: Search using MCP tool with configurable limits."""
            search_tool = await get_search_tool_for_server(state["mcp_server"])
            if not search_tool:
                return {"mcp_context": f"No search tool available for {state['mcp_server']}"}
            try:
                # Get configured search parameters for this server
                base_params = {"query": state["prompt"]}
                server_config = get_mcp_search_config(state["mcp_server"])
                search_params = {**base_params, **server_config}
                
                print(f"🔍 MCP Search ({state['mcp_server']}): {search_params}")
                
                result = await search_tool.ainvoke(search_params)
                result_str = str(result)
                
                # Check if truncation is needed
                was_truncated, final_context = should_truncate_context(result_str)
                
                if was_truncated:
                    print(f"⚠️  Context truncated from {len(result_str)} to {len(final_context)} chars")
                
                return {"mcp_context": final_context}
            except Exception as e:
                return {"mcp_context": f"Error executing search: {str(e)}"}

        def generate_mcp_answer(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """MCP Branch Step 2: Generate answer using MCP context."""
            llm = create_llm(state["model_name"])
            prompt_template = f"""Based on the following web search results, answer the question: {state["prompt"]}

Context (from web search):
{state["mcp_context"]}

Please provide a comprehensive answer based on the available information."""
            
            # Capture response and cost information
            response, cost = llm.invoke(prompt_template)
            
            return {
                "mcp_answer": response.content,
                "mcp_generation_cost": cost
            }

        def evaluate_mcp(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """MCP Branch Step 3: Evaluate MCP answer with RAGAS."""
            contexts = [state["mcp_context"]] if state["mcp_context"] else []
            metrics = self.evaluator.evaluate_response(
                question=state["prompt"],
                answer=state["mcp_answer"],
                contexts=contexts
            )
            return {"mcp_metrics": metrics}

        workflow = StateGraph(UnifiedWorkflowState)

        workflow.add_node("retrieve_rag", retrieve_rag_context)
        workflow.add_node("generate_rag", generate_rag_answer)
        workflow.add_node("evaluate_rag", evaluate_rag)

        workflow.add_node("search_mcp", search_mcp_context)
        workflow.add_node("generate_mcp", generate_mcp_answer)
        workflow.add_node("evaluate_mcp", evaluate_mcp)

        workflow.add_edge(START, "retrieve_rag")
        workflow.add_edge(START, "search_mcp")

        workflow.add_edge("retrieve_rag", "generate_rag")
        workflow.add_edge("generate_rag", "evaluate_rag")
        workflow.add_edge("evaluate_rag", END)

        workflow.add_edge("search_mcp", "generate_mcp")
        workflow.add_edge("generate_mcp", "evaluate_mcp")
        workflow.add_edge("evaluate_mcp", END)

        return workflow.compile()


async def execute_unified_workflow(
    prompt: str,
    model_name: str,
    rag_type: str,
    mcp_server: str
) -> Dict[str, Any]:
    """
    Execute the unified workflow with parallel RAG and MCP branches.

    Args:
        prompt: User question
        model_name: LLM model to use ("gpt-5" or "gemini-2.5-pro")
        rag_type: Type of RAG ("naive" or "hybrid")
        mcp_server: MCP server to use ("tavily" or "duckduckgo")

    Returns:
        Single unified JSON with both RAG and MCP results
    """
    unified_workflow = UnifiedWorkflow()
    workflow = unified_workflow.create_workflow()

    initial_state: UnifiedWorkflowState = {
        "execution_id": str(uuid.uuid4()),
        "prompt": prompt,
        "model_name": model_name,
        "rag_type": rag_type,
        "mcp_server": mcp_server,
        "timestamp": datetime.now().isoformat(),
        "rag_context": [],
        "rag_answer": "",
        "rag_metrics": {},
        "rag_generation_cost": {},  # Initialize cost tracking
        "mcp_context": "",
        "mcp_answer": "",
        "mcp_metrics": {},
        "mcp_generation_cost": {}  # Initialize cost tracking
    }

    result = await workflow.ainvoke(initial_state)

    # Calculate total costs from both branches
    rag_cost = result.get("rag_generation_cost", {}).get("total_cost_usd", 0.0)
    mcp_cost = result.get("mcp_generation_cost", {}).get("total_cost_usd", 0.0)
    total_cost = rag_cost + mcp_cost

    output = {
        "execution_id": result["execution_id"],
        "timestamp": result["timestamp"],
        "configuration": {
            "model": model_name,
            "rag_type": rag_type,
            "mcp_server": mcp_server
        },
        "prompt": result["prompt"],
        "rag_results": {
            "retrieved_context": result["rag_context"],
            "generated_answer": result["rag_answer"],
            "ragas_metrics": result["rag_metrics"],
            "cost_details": result.get("rag_generation_cost", {})  # Add cost details
        },
        "mcp_results": {
            "retrieved_context": [result["mcp_context"]],
            "generated_answer": result["mcp_answer"],
            "ragas_metrics": result["mcp_metrics"],
            "cost_details": result.get("mcp_generation_cost", {})  # Add cost details
        },
        "cost_summary": {
            "rag_cost_usd": round(rag_cost, 8),
            "mcp_cost_usd": round(mcp_cost, 8),
            "total_cost_usd": round(total_cost, 8),
            "model_used": model_name
        }
    }

    return output
