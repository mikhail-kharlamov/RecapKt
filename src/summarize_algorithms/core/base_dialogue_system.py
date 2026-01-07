import functools
import os

from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from src.benchmarking.memory_logger import MemoryLogger
from src.summarize_algorithms.core.dialogue import Dialogue
from src.summarize_algorithms.core.graph_nodes import (
    UpdateState,
    generate_response_node,
    should_continue_memory_update,
    update_memory_node,
)
from src.summarize_algorithms.core.models import (
    DialogueState,
    OpenAIModels,
    Session,
    WorkflowNode,
)
from src.summarize_algorithms.core.prompts import RESPONSE_GENERATION_PROMPT
from src.summarize_algorithms.core.response_generator import ResponseGenerator


class BaseDialogueSystem(ABC, Dialogue):
    def __init__(
        self,
        llm: BaseChatModel | None = None,
        embed_code: bool = False,
        embed_tool: bool = False,
        embed_model: Embeddings | None = None,
        max_session_id: int = 3,
        system_name: str | None = None
    ) -> None:
        if system_name is None:
            self.system_name = self.__class__.__name__
        else:
            self.system_name = system_name

        load_dotenv()

        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            self.llm = llm or ChatOpenAI(
                model=OpenAIModels.GPT_4_O_MINI.value,
                api_key=SecretStr(api_key)
            )
        else:
            raise ValueError("OPENAI_API_KEY environment variable is not loaded")

        self.summarizer = self._build_summarizer()
        self.graph = self._build_graph()
        self.state: DialogueState | None = None
        self.embed_code = embed_code
        self.embed_tool = embed_tool
        self.embed_model = embed_model
        self.max_session_id = max_session_id
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0

        self.memory_logger = MemoryLogger()
        self.iteration = 0

    @abstractmethod
    def _build_summarizer(self) -> Any:
        pass

    @staticmethod
    def _get_response_prompt_template() -> PromptTemplate:
        return RESPONSE_GENERATION_PROMPT

    @abstractmethod
    def _get_initial_state(self, sessions: list[Session], last_session: Session, query: str) -> DialogueState:
        pass

    @property
    @abstractmethod
    def _get_dialogue_state_class(self) -> type[DialogueState]:
        pass

    def _build_graph(
            self,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None,
            system_prompt_template: str = ""
    ) -> CompiledStateGraph:

        safe_system_prompt = system_prompt_template.replace("{", "{{").replace("}", "}}")
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", safe_system_prompt),
            MessagesPlaceholder("history")
        ])

        self.response_generator = ResponseGenerator(
            self.llm,
            chat_prompt,
            structure,
            tools
        )

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

    def process_dialogue(
            self,
            sessions: list[Session],
            system_prompt: str,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None
    ) -> DialogueState:
        graph = self._build_graph(structure, tools, system_prompt_template=system_prompt)
        initial_state = self._get_initial_state(sessions[:-1], sessions[-1], system_prompt)

        with get_openai_callback() as cb:
            result_state = graph.invoke(initial_state)
            self.state = self._get_dialogue_state_class(**result_state)

            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens
            self.total_cost += cb.total_cost

        return self.state if self.state is not None else initial_state
