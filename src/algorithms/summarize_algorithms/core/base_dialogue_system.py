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
from langchain_ollama.chat_models import ChatOllama
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
    LocalModels,
    OpenAIModels,
    Session,
    WorkflowNode, MemoryDialogueState,
)
from src.summarize_algorithms.core.prompts import RESPONSE_GENERATION_PROMPT
from src.summarize_algorithms.core.response_generator import ResponseGenerator


class BaseDialogueSystem(ABC, Dialogue):
    """
    Shared LangGraph-based implementation for dialogue systems in this repository.

    The pipeline is built as a graph with two main stages:
    1) update memory (via a concrete `BaseSummarizer` implementation)
    2) generate the final response (via `ResponseGenerator`, optionally with tools/structured output)

    Subclasses plug in the summarizer and the initial `DialogueState`.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        embed_code: bool = False,
        embed_tool: bool = False,
        embed_model: Embeddings | None = None,
        max_session_id: int = 3,
        system_name: str | None = None,
        is_local: bool = True,
    ) -> None:
        self.system_name = system_name or self.__class__.__name__

        self._initialize_model(llm, is_local)

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

    def _initialize_model(self, llm: BaseChatModel | None = None, is_local: bool = False) -> None:
        load_dotenv()

        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            self.memory_llm = ChatOpenAI(
                model=OpenAIModels.GPT_5_MINI.value,
                api_key=SecretStr(api_key)
            )
        else:
            raise ValueError("OPENAI_API_KEY environment variable is not loaded")

        if is_local:
            self.llm = ChatOllama(
                model=LocalModels.QWEN_2_5_14_B.value,
                temperature=0,
                keep_alive="1h"
            )
        else:
            self.llm = llm or ChatOpenAI(
                model=OpenAIModels.GPT_4_O_MINI.value,
                api_key=SecretStr(api_key)
            )

    def _build_graph(
            self,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None,
    ) -> CompiledStateGraph:
        chat_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("history")
        ])

        self.response_generator = ResponseGenerator(
            self.llm,
            chat_prompt,
            structure=structure,
            tools=tools,
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
        """
        Run the dialogue workflow and return the final `DialogueState`.

        :param sessions: past user/assistant/tool interactions (last element is treated as the current session).
        :param system_prompt: system prompt template used during response generation.
        :param structure: optional JSON schema for structured model output.
        :param tools: optional tools/functions specs for tool calling.
        :return: DialogueState: state populated with updated memory and the generated response.
        """
        graph = self._build_graph(structure, tools)
        initial_state = self._get_initial_state(sessions, sessions[-1], system_prompt)

        with get_openai_callback() as cb:
            result_state = graph.invoke(initial_state)
            self.state = self._get_dialogue_state_class(**result_state)

            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens
            self.total_cost += cb.total_cost

        return self.state if self.state is not None else initial_state
