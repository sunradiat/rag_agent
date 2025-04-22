from typing import Dict, List, TypedDict, Annotated, Any
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import networkx as nx
import matplotlib.pyplot as plt

from .query_processor import QueryProcessor
from .query_decomposer import QueryDecomposer
from .retriever import Retriever
from .reranker import Reranker
from .generation import Generator

class QueryState(TypedDict):
    """State for query processing pipeline"""
    original_query: str
    rewritten_query: str
    is_complex: bool
    sub_questions: List[str]
    current_step: str
    explanation: str
    retrieved_docs: List[Dict]
    reranked_docs: List[Dict]
    sub_answers: List[Dict]
    final_answer: str
    sources: List[str]
    conversation_history: List[Dict]  # Add conversation history
    company: str
    service: str

class QueryProcessorGraph:
    def __init__(self, input_folder: str):
        """Initialize the query processing graph with all necessary components"""
        self.query_processor = QueryProcessor()
        self.query_decomposer = QueryDecomposer()
        self.retriever = Retriever(input_folder)
        self.reranker = Reranker()
        self.generator = Generator()
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> Graph:
        """Create the query processing graph"""
        # 1. Define the state graph
        workflow = StateGraph(QueryState)
        
        # 2. Define node functions
        def rewrite_query(state: QueryState) -> QueryState:
            """Rewrite the query for better understanding"""
            # Include conversation history in the query rewriting
            history_context = ""
            if state["conversation_history"]:
                history_context = "\nPrevious conversation:\n"
                for msg in state["conversation_history"][-3:]:  # Use last 3 messages
                    history_context += f"Q: {msg['query']}\nA: {msg['answer']}\n"
            
            result = self.query_processor.rewrite(
                query=state["original_query"],
                context=history_context
            )
            state["rewritten_query"] = result["main_query"]
            state["current_step"] = "query_rewriting"
            return state
        
        def analyze_complexity(state: QueryState) -> QueryState:
            """Analyze if the query is complex"""
            result = self.query_decomposer.analyze_complexity(state["rewritten_query"])
            state["is_complex"] = result["is_complex"]
            state["explanation"] = result["explanation"]
            state["current_step"] = "complexity_analysis"
            return state
        
        def decompose_query(state: QueryState) -> QueryState:
            """Decompose complex query into sub-questions"""
            if not state["is_complex"]:
                return state
                
            result = self.query_decomposer.decompose(state["rewritten_query"])
            state["sub_questions"] = result["sub_questions"]
            state["current_step"] = "query_decomposition"
            return state
        
        def retrieve_documents(state: QueryState) -> QueryState:
            """Retrieve relevant documents"""
            if state["is_complex"]:
                # For complex queries, retrieve for each sub-question
                all_results = []
                seen_docs = set()
                
                for sub_query in state["sub_questions"]:
                    sub_results = self.retriever.retrieve(
                        query=sub_query,
                        n_results=5,
                        company=state.get("company"),
                        service=state.get("service")
                    )
                    
                    # Add unique results
                    for result in sub_results:
                        doc_id = result['metadata']['file_name']
                        if doc_id not in seen_docs:
                            seen_docs.add(doc_id)
                            all_results.append(result)
            else:
                # For simple queries, retrieve using main query
                all_results = self.retriever.retrieve(
                    query=state["rewritten_query"],
                    n_results=5,
                    company=state.get("company"),
                    service=state.get("service")
                )
            
            state["retrieved_docs"] = all_results
            state["current_step"] = "document_retrieval"
            return state
        
        def rerank_documents(state: QueryState) -> QueryState:
            """Rerank retrieved documents"""
            reranked_results = self.reranker.rerank(
                query=state["rewritten_query"],
                results=state["retrieved_docs"],
                top_k=min(len(state["retrieved_docs"]), 5)
            )
            
            state["reranked_docs"] = reranked_results
            state["current_step"] = "document_reranking"
            return state
        
        def generate_sub_answers(state: QueryState) -> QueryState:
            """Generate answers for sub-questions"""
            if not state["is_complex"]:
                return state
                
            sub_answers = []
            for sub_query in state["sub_questions"]:
                answer = self.generator.generate_answer_with_sources(
                    query=sub_query,
                    results=state["reranked_docs"]
                )
                sub_answers.append({
                    "question": sub_query,
                    "answer": answer["answer"],
                    "sources": answer["sources"]
                })
            
            state["sub_answers"] = sub_answers
            state["current_step"] = "sub_answer_generation"
            return state
        
        def generate_final_answer(state: QueryState) -> QueryState:
            """Generate final answer"""
            if state["is_complex"]:
                # For complex queries, combine sub-answers
                answer = self.generator.combine_answers(
                    original_query=state["original_query"],
                    sub_answers=state["sub_answers"]
                )
            else:
                # For simple queries, generate directly
                answer = self.generator.generate_answer_with_sources(
                    query=state["rewritten_query"],
                    results=state["reranked_docs"]
                )
            
            state["final_answer"] = answer["answer"]
            state["sources"] = answer["sources"]
            state["current_step"] = "final_answer_generation"
            
            # Update conversation history
            if "conversation_history" not in state:
                state["conversation_history"] = []
            state["conversation_history"].append({
                "query": state["original_query"],
                "answer": state["final_answer"],
                "sources": state["sources"]
            })
            
            return state
        
        # 3. Add nodes
        workflow.add_node("rewrite_query", rewrite_query)
        workflow.add_node("analyze_complexity", analyze_complexity)
        workflow.add_node("decompose_query", decompose_query)
        workflow.add_node("retrieve_documents", retrieve_documents)
        workflow.add_node("rerank_documents", rerank_documents)
        workflow.add_node("generate_sub_answers", generate_sub_answers)
        workflow.add_node("generate_final_answer", generate_final_answer)
        
        # 4. Define edges
        workflow.add_edge("rewrite_query", "analyze_complexity")
        workflow.add_conditional_edges(
            "analyze_complexity",
            lambda x: "decompose_query" if x["is_complex"] else "retrieve_documents",
            ["decompose_query", "retrieve_documents"]
        )
        workflow.add_edge("decompose_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "rerank_documents")
        workflow.add_conditional_edges(
            "rerank_documents",
            lambda x: "generate_sub_answers" if x["is_complex"] else "generate_final_answer",
            ["generate_sub_answers", "generate_final_answer"]
        )
        workflow.add_edge("generate_sub_answers", "generate_final_answer")
        
        # 5. Set entry point
        workflow.set_entry_point("rewrite_query")
        
        # 6. Compile the graph
        return workflow.compile()
    
    def process_query(self, query: str, company: str = None, service: str = None) -> Dict[str, Any]:
        """Process a query through the pipeline"""
        # Initialize state
        initial_state = {
            "original_query": query,
            "rewritten_query": "",
            "is_complex": False,
            "sub_questions": [],
            "current_step": "",
            "explanation": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "sub_answers": [],
            "final_answer": "",
            "sources": [],
            "conversation_history": [],
            "company": company,
            "service": service
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Return the result
        return {
            "query": final_state["original_query"],
            "rewritten_query": final_state["rewritten_query"],
            "is_complex": final_state["is_complex"],
            "explanation": final_state["explanation"],
            "answer": final_state["final_answer"],
            "sources": final_state["sources"],
            "sub_questions": final_state["sub_questions"] if final_state["is_complex"] else None,
            "sub_answers": final_state["sub_answers"] if final_state["is_complex"] else None,
            "conversation_history": final_state["conversation_history"]
        }
    
    def visualize_graph(self, output_path: str = "query_processor_graph.png"):
        """Visualize the query processing graph"""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.graph.nodes:
            G.add_node(node)
        
        # Add edges
        for edge in self.graph.edges:
            G.add_edge(edge[0], edge[1])
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Save the plot
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization saved to {output_path}") 