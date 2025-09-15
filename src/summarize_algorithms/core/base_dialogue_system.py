import functools

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from langchain_community.callbacks import get_openai_callback
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.summarize_algorithms.core.graph_nodes import (
    UpdateState,
    generate_response_node,
    should_continue_memory_update,
    update_memory_node,
)
from src.summarize_algorithms.core.models import DialogueState, Session, WorkflowNode
from src.summarize_algorithms.core.prompts import RESPONSE_GENERATION_PROMPT
from src.summarize_algorithms.core.response_generator import ResponseGenerator


class BaseDialogueSystem(ABC):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        embed_code: bool = False,
        embed_tool: bool = False,
        embed_model: Optional[Embeddings] = None,
        max_session_id: int = 3,
    ) -> None:
        self.llm = llm or ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
        self.summarizer = self._build_summarizer()
        self.response_generator = ResponseGenerator(
            self.llm, self._get_response_prompt_template()
        )
        self.graph = self._build_graph()
        self.state: Optional[DialogueState] = None
        self.embed_code = embed_code
        self.embed_tool = embed_tool
        self.embed_model = embed_model
        self.max_session_id = max_session_id
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0

    @abstractmethod
    def _build_summarizer(self) -> Any:
        pass

    @staticmethod
    def _get_response_prompt_template() -> PromptTemplate:
        return RESPONSE_GENERATION_PROMPT

    @abstractmethod
    def _get_initial_state(self, sessions: list[Session], query: str) -> DialogueState:
        pass

    @property
    @abstractmethod
    def _get_dialogue_state_class(self) -> Type[DialogueState]:
        pass

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(self._get_dialogue_state_class)

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
                UpdateState.CONTINUE_UPDATE.value: WorkflowNode.UPDATE_MEMORY.value,
                UpdateState.FINISH_UPDATE.value: WorkflowNode.GENERATE_RESPONSE.value,
            },
        )

        workflow.add_edge(WorkflowNode.GENERATE_RESPONSE.value, END)

        return workflow.compile()

    def process_dialogue(self, sessions: list[Session], query: str) -> DialogueState:
        initial_state = self._get_initial_state(sessions, query)
        with get_openai_callback() as cb:
            self.state = self._get_dialogue_state_class(
                **self.graph.invoke(initial_state)
            )

            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens
            self.total_cost += cb.total_cost
        return self.state if self.state is not None else initial_state
