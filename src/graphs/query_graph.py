# src/graphs/query_graph.py
"""
Query Execution Graph - LangGraph State Machine

Implements the query flow as a stateful graph:
1. Parse → 2. Search PageIndex → 3. Retrieve LDUs → 4. Synthesize → 5. Cite → 6. Log

This enables:
- Conditional escalation (low confidence → retry with VLM)
- Caching of intermediate results
- Parallel retrieval of multiple sections
- Audit trail at each step
"""

from typing import TypedDict, Annotated, Literal, Sequence
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.query_agent import QueryAgent, QueryState


class QueryGraph:
    """LangGraph-based query execution workflow."""
    
    def __init__(self, agent: QueryAgent):
        """Initialize graph with QueryAgent instance."""
        self.agent = agent
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("parse_query", self._parse_query)
        workflow.add_node("search_pageindex", self._search_pageindex_node)
        workflow.add_node("retrieve_ldus", self._retrieve_ldus_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
        workflow.add_node("build_citations", self._build_citations_node)
        workflow.add_node("log_audit", self._log_audit_node)
        
        # Define edges
        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query", "search_pageindex")
        workflow.add_edge("search_pageindex", "retrieve_ldus")
        workflow.add_edge("retrieve_ldus", "synthesize_answer")
        
        # Conditional edge: escalate if confidence low
        workflow.add_conditional_edges(
            "synthesize_answer",
            self._should_escalate,
            {
                "escalate": "synthesize_answer",  # Retry with VLM strategy
                "continue": "build_citations"
            }
        )
        
        workflow.add_edge("build_citations", "log_audit")
        workflow.add_edge("log_audit", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def _parse_query(self, state: QueryState) -> QueryState:
        """Parse query and extract keywords."""
        keywords = self.agent._extract_query_keywords(state.query)
        # Store in state for downstream nodes
        state.retrieval_metadata = {"keywords": keywords}  # type: ignore
        return state
    
    def _search_pageindex_node(self, state: QueryState) -> QueryState:
        """Search PageIndex for relevant sections."""
        # This would load pageindex and search
        # For now, placeholder
        return state
    
    def _retrieve_ldus_node(self, state: QueryState) -> QueryState:
        """Retrieve LDUs from matched sections."""
        # Placeholder for LDU retrieval
        return state
    
    def _synthesize_answer_node(self, state: QueryState) -> QueryState:
        """Synthesize answer from retrieved LDUs."""
        # Placeholder for answer synthesis
        state.answer = "Synthesized answer"
        state.confidence = 0.85
        return state
    
    def _should_escalate(self, state: QueryState) -> Literal["escalate", "continue"]:
        """Decide whether to escalate to VLM strategy."""
        if state.confidence < 0.75 and state.strategy_used != "vision_augmented":
            state.strategy_used = "vision_augmented"
            return "escalate"
        return "continue"
    
    def _build_citations_node(self, state: QueryState) -> QueryState:
        """Build ProvenanceCitations from retrieved LDUs."""
        # Placeholder for citation building
        return state
    
    def _log_audit_node(self, state: QueryState) -> QueryState:
        """Log query to audit ledger."""
        # Placeholder for audit logging
        return state
    
    def invoke(self, query: str, doc_id: str, config: dict = None) -> QueryState:
        """Execute query through the graph."""
        initial_state = QueryState(query=query, doc_id=doc_id)
        return self.graph.invoke(initial_state, config=config)
