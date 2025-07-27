import functools

from enum import Enum
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.recsum.graph_nodes import (
    generate_response_node,
    should_continue_memory_update,
    update_memory_node,
)
from src.recsum.models import DialogueState, Session
from src.recsum.prompts import (
    MEMORY_UPDATE_PROMPT_TEMPLATE,
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
)
from src.recsum.response_generator import ResponseGenerator
from src.recsum.summarizer import RecursiveSummarizer


class WorkflowNode(Enum):
    UPDATE_MEMORY = "update_memory"
    GENERATE_RESPONSE = "generate_response"


class ConditionalEdge(Enum):
    CONTINUE_UPDATE = "continue_update"
    FINISH_UPDATE = "finish_update"


class DialogueSystem:
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm or ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
        self.summarizer = RecursiveSummarizer(self.llm, MEMORY_UPDATE_PROMPT_TEMPLATE)
        self.response_generator = ResponseGenerator(
            self.llm, RESPONSE_GENERATION_PROMPT_TEMPLATE
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(DialogueState)

        workflow.add_node(
            WorkflowNode.UPDATE_MEMORY.value,
            functools.partial(update_memory_node, self.summarizer),
        )
        workflow.add_node(
            WorkflowNode.GENERATE_RESPONSE.value,
            functools.partial(generate_response_node, self.response_generator),
        )

        workflow.set_entry_point(WorkflowNode.UPDATE_MEMORY.value)

        workflow.add_conditional_edges(
            WorkflowNode.UPDATE_MEMORY.value,
            should_continue_memory_update,
            {
                ConditionalEdge.CONTINUE_UPDATE.value: WorkflowNode.UPDATE_MEMORY.value,
                ConditionalEdge.FINISH_UPDATE.value: WorkflowNode.GENERATE_RESPONSE.value,
            },
        )

        workflow.add_edge(WorkflowNode.GENERATE_RESPONSE.value, END)

        return workflow.compile()

    def process_dialogue(self, sessions: list[Session], query: str) -> DialogueState:
        initial_state = DialogueState(
            dialogue_sessions=sessions,
            current_session_index=0,
            query=query,
            response="",
        )

        return DialogueState(**self.graph.invoke(initial_state))
