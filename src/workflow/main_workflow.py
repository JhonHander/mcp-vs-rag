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

    mcp_context: str
    mcp_answer: str
    mcp_metrics: Dict[str, float]


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
            response = llm.invoke(prompt_template)
            return {"rag_answer": response.content}

        def evaluate_rag(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """RAG Branch Step 3: Evaluate RAG answer with RAGAS."""
            metrics = self.evaluator.evaluate_response(
                question=state["prompt"],
                answer=state["rag_answer"],
                contexts=state["rag_context"]
            )
            return {"rag_metrics": metrics}

        async def search_mcp_context(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """MCP Branch Step 1: Search using MCP tool."""
            search_tool = await get_search_tool_for_server(state["mcp_server"])
            if not search_tool:
                return {"mcp_context": f"No search tool available for {state['mcp_server']}"}
            try:
                result = await search_tool.ainvoke({"query": state["prompt"]})
                return {"mcp_context": str(result)}
            except Exception as e:
                return {"mcp_context": f"Error executing search: {str(e)}"}

        def generate_mcp_answer(state: UnifiedWorkflowState) -> Dict[str, Any]:
            """MCP Branch Step 2: Generate answer using MCP context."""
            llm = create_llm(state["model_name"])
            prompt_template = f"""Based on the following web search results, answer the question: {state["prompt"]}

Context (from web search):
{state["mcp_context"]}

Please provide a comprehensive answer based on the available information."""
            response = llm.invoke(prompt_template)
            return {"mcp_answer": response.content}

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
        "mcp_context": "",
        "mcp_answer": "",
        "mcp_metrics": {}
    }

    result = await workflow.ainvoke(initial_state)

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
            "ragas_metrics": result["rag_metrics"]
        },
        "mcp_results": {
            "retrieved_context": [result["mcp_context"]],
            "generated_answer": result["mcp_answer"],
            "ragas_metrics": result["mcp_metrics"]
        }
    }

    return output
