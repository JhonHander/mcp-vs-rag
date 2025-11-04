"""
Main workflow orchestration using LangGraph.
Implements parallel RAG and MCP branches that merge into a single unified JSON output.
Both pipelines execute independently but results are combined for comparison.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
import uuid

from src.rag.rag_implementations import NaiveRAG, HybridRAG
from src.mcp.mcp_client import get_search_tool_for_server
from src.models.llm_factory import create_llm
from src.evaluation.ragas_evaluator import RAGASEvaluator


class UnifiedWorkflowState(BaseModel):
    """
    Unified state for parallel RAG and MCP execution.
    Both branches populate their own fields independently.
    """
    # Common fields
    execution_id: str
    prompt: str
    model_name: str
    rag_type: str
    mcp_server: str
    timestamp: str = ""

    # RAG branch results
    rag_context: List[str] = []
    rag_answer: str = ""
    rag_metrics: Dict[str, float] = {}

    # MCP branch results
    mcp_context: str = ""
    mcp_answer: str = ""
    mcp_metrics: Dict[str, float] = {}


class UnifiedWorkflow:
    """
    Unified workflow with parallel RAG and MCP branches.

    Flow:
        START
          ↓
        ┌─┴─┐
        │   │
      RAG  MCP
        │   │
        └─┬─┘
          ↓
        MERGE
          ↓
         END
    """

    def __init__(self):
        self.naive_rag = NaiveRAG()
        self.hybrid_rag = HybridRAG()
        self.evaluator = RAGASEvaluator()

    def create_workflow(self) -> StateGraph:
        """Create the unified workflow with parallel branches."""

        # ========== RAG BRANCH ==========
        def retrieve_rag_context(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """RAG Branch Step 1: Retrieve context from RAG."""
            if state.rag_type == "naive":
                contexts = self.naive_rag.retrieve(state.prompt)
            else:  # hybrid
                contexts = self.hybrid_rag.retrieve(state.prompt)

            return {"rag_context": contexts}

        def generate_rag_answer(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """RAG Branch Step 2: Generate answer using RAG context."""
            llm = create_llm(state.model_name)

            rag_text = "\n".join(state.rag_context)
            prompt_template = f"""Based on the following context, answer the question: {state.prompt}

Context (from knowledge base):
{rag_text}

Please provide a comprehensive answer based on the available information."""

            response = llm.invoke(prompt_template)
            return {"rag_answer": response.content}

        def evaluate_rag(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """RAG Branch Step 3: Evaluate RAG answer with RAGAS."""
            metrics = self.evaluator.evaluate_response(
                question=state.prompt,
                answer=state.rag_answer,
                contexts=state.rag_context
            )
            return {"rag_metrics": metrics}

        # ========== MCP BRANCH ==========
        async def search_mcp_context(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """MCP Branch Step 1: Search using MCP tool."""
            search_tool = await get_search_tool_for_server(state.mcp_server)

            if not search_tool:
                return {"mcp_context": f"No search tool available for {state.mcp_server}"}

            try:
                result = await search_tool.ainvoke({"query": state.prompt})
                return {"mcp_context": str(result)}
            except Exception as e:
                return {"mcp_context": f"Error executing search: {str(e)}"}

        def generate_mcp_answer(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """MCP Branch Step 2: Generate answer using MCP context."""
            llm = create_llm(state.model_name)

            prompt_template = f"""Based on the following web search results, answer the question: {state.prompt}

Context (from web search):
{state.mcp_context}

Please provide a comprehensive answer based on the available information."""

            response = llm.invoke(prompt_template)
            return {"mcp_answer": response.content}

        def evaluate_mcp(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """MCP Branch Step 3: Evaluate MCP answer with RAGAS."""
            contexts = [state.mcp_context] if state.mcp_context else []

            metrics = self.evaluator.evaluate_response(
                question=state.prompt,
                answer=state.mcp_answer,
                contexts=contexts
            )
            return {"mcp_metrics": metrics}

        # ========== PARALLEL EXECUTION ==========
        async def execute_parallel_branches(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """
            Execute both RAG and MCP branches in parallel.
            This node orchestrates both pipelines simultaneously.
            """
            # Execute RAG branch
            async def run_rag_branch():
                # Step 1: Retrieve
                rag_ctx = retrieve_rag_context(state)
                state.rag_context = rag_ctx["rag_context"]

                # Step 2: Generate answer
                rag_ans = generate_rag_answer(state)
                state.rag_answer = rag_ans["rag_answer"]

                # Step 3: Evaluate
                rag_met = evaluate_rag(state)
                return rag_met["rag_metrics"]

            # Execute MCP branch
            async def run_mcp_branch():
                # Step 1: Search
                mcp_ctx = await search_mcp_context(state)
                state.mcp_context = mcp_ctx["mcp_context"]

                # Step 2: Generate answer
                mcp_ans = generate_mcp_answer(state)
                state.mcp_answer = mcp_ans["mcp_answer"]

                # Step 3: Evaluate
                mcp_met = evaluate_mcp(state)
                return mcp_met["mcp_metrics"]

            # Run both branches in parallel
            rag_metrics, mcp_metrics = await asyncio.gather(
                run_rag_branch(),
                run_mcp_branch()
            )

            return {
                "rag_context": state.rag_context,
                "rag_answer": state.rag_answer,
                "rag_metrics": rag_metrics,
                "mcp_context": state.mcp_context,
                "mcp_answer": state.mcp_answer,
                "mcp_metrics": mcp_metrics
            }

        # ========== BUILD GRAPH ==========
        workflow = StateGraph(UnifiedWorkflowState)

        # Single node that executes both branches in parallel
        workflow.add_node("parallel_execution", execute_parallel_branches)

        # Simple linear flow
        workflow.add_edge(START, "parallel_execution")
        workflow.add_edge("parallel_execution", END)

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
    # Create workflow
    unified_workflow = UnifiedWorkflow()
    workflow = unified_workflow.create_workflow()

    # Initialize state
    initial_state = UnifiedWorkflowState(
        execution_id=str(uuid.uuid4()),
        prompt=prompt,
        model_name=model_name,
        rag_type=rag_type,
        mcp_server=mcp_server,
        timestamp=datetime.now().isoformat()
    )

    # Execute workflow (both branches run in parallel)
    result = await workflow.ainvoke(initial_state)

    # Format unified output
    output = {
        "execution_id": result.execution_id,
        "timestamp": result.timestamp,
        "configuration": {
            "model": model_name,
            "rag_type": rag_type,
            "mcp_server": mcp_server
        },
        "prompt": result.prompt,

        # RAG results
        "rag_results": {
            "retrieved_context": result.rag_context,
            "generated_answer": result.rag_answer,
            "ragas_metrics": result.rag_metrics
        },

        # MCP results
        "mcp_results": {
            "retrieved_context": [result.mcp_context],
            "generated_answer": result.mcp_answer,
            "ragas_metrics": result.mcp_metrics
        }
    }

    return output
