from typing import Annotated, TypedDict

from app.config import get_settings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable


class AgentState(TypedDict):
    """State for the production agent
    Uses Anotated with add_messages reducer for message accumulation
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: str | None
    retry_count: int
    model_used: str


class ProductionAgent:
    """
    Production LangGraph agent with:
    - Retry on failure (model fallback)
    - Graceful error handling
    - Langsmith tracing
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatOllama(model=settings.primary_model, temperature=0)
        self.fallback_llm = ChatOllama(model=settings.fallback_model, temperature=0)
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build LangGraph state machine"""

        def process_message(state: AgentState) -> dict:
            """Try to process the messaage with the primary model"""

            try:
                response = self.primary_llm.invoke(input=state["messages"])
                return {"messages": [response], "error": None, "model_used": "primary"}
            except Exception as ex:
                return {
                    "error": str(ex),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        def try_fallback(state: AgentState) -> dict:
            """Fall back to secondary model"""

            try:
                response = self.fallback_llm.invoke(input=state["messages"])
                return {"messages": [response], "error": None, "model_used": "fallback"}
            except Exception as ex:
                return {"error": str(ex), "model_used": ""}

        def handle_error(state: AgentState) -> dict:
            """Return a gracefull error message"""

            return {
                "messages": AIMessage(
                    content="I'm sorry, I'm having trouble processing your request "
                    "right now. Please try again in a moment."
                ),
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """Decide what to do after primary model attempt"""

            if state.get("error") is None:
                return "done"
            elif state.get("retry_count") < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """Decide what to do after fallback attempts"""
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        # ! Build the graph

        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")

        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )

        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {
                "done": END,
                "error": "error",
            },
        )

        graph.add_edge("error", END)

        return graph.compile()

    def invoke(self, message: str) -> dict:
        """Invoke the agent with a user message.


        Args:
            message (str): _description_

        Returns:
            dict: {"response": str, "model_used": str, "error": str | None}
        """

        result = self.graph.invoke(
            input={
                "messages": [HumanMessage(content=message)],
                "error": None,
                "retry_count": 0,
                "model_used": "",
            }
        )

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }
