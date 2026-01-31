import logging
import os

from typing import Any

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.benchmark.logger.baseline_logger import BaselineLogger
from src.utils.system_prompt_builder import MemorySections, SystemPromptBuilder
from src.algorithms.dialogue import Dialogue
from src.algorithms.summarize_algorithms.core.models import (
    DialogueState,
    LocalModels,
    OpenAIModels,
    Session,
)


class DialogueBaseline(Dialogue):
    """
    Baseline dialogue system that answers using the full (compressed) conversation context.

    This implementation does not build or retrieve long-term memory. It simply converts all provided `Session`s into
    a single message history, crops it to a token budget, and calls an LLM.
    """

    def __init__(self, system_name: str, llm: BaseChatModel | None = None, is_local: bool = True) -> None:
        load_dotenv()

        self.system_name = system_name

        self._initialize_model(llm, is_local)

        self._prompt_builder = SystemPromptBuilder()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0

        self.baseline_logger = BaselineLogger()

    def _initialize_model(self, llm: BaseChatModel | None = None, is_local: bool = False) -> None:
        """
        Initialize the underlying chat model.

        :param llm: optional externally constructed model instance.
        :param is_local: if True, uses an Ollama model; otherwise uses OpenAI.
        :return: None
        """
        if is_local:
            self.llm = ChatOllama(
                model=LocalModels.QWEN_2_5_14_B.value,
                temperature=0.7,
                keep_alive="1h"
            )
            return

        load_dotenv()

        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            self.llm = llm or ChatOpenAI(
                model=OpenAIModels.GPT_4_O_MINI.value,
                api_key=SecretStr(api_key)
            )
        else:
            raise ValueError("OPENAI_API_KEY environment variable is not loaded")

    def _build_chain(
            self,
            prompt: ChatPromptTemplate,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None
    ) -> Runnable:
        """
        Build a LangChain runnable for the current run.

        Depending on `structure`/`tools`, the chain may return:
        - plain text
        - structured JSON
        - tool calls

        :param prompt: chat prompt template (system + history placeholder).
        :param structure: optional schema for structured output.
        :param tools: optional tool specs for tool calling.
        :return: Runnable: composed chain.
        """

        if structure and not tools:
            structured_llm = self.llm.with_structured_output(structure)
            return prompt | structured_llm

        if tools and not structure:
            llm_with_tools = self.llm.bind_tools(tools)
            return prompt | llm_with_tools

        if tools and structure:
            tools = [DialogueBaseline._get_return_action_plan(structure), *tools]
            llm_with_tools = self.llm.bind_tools(tools)
            return prompt | llm_with_tools

        return prompt | self.llm | StrOutputParser()

    @staticmethod
    def _get_return_action_plan(structure: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "return_action_plan",
                "description": "Return the final JSON action plan strictly matching the schema.",
                "parameters": structure,
            },
        }

    @staticmethod
    def _get_context(sessions: list[Session]) -> list[BaseMessage]:
        context_messages: list[BaseMessage] = []
        for session in sessions:
            context_messages.extend(session.to_langchain_messages())
        return context_messages

    def _crop(self, messages: list[Any], max_tokens: int = 100000) -> list[BaseMessage]:
        """
        Trim a list of messages to fit into a token budget.

        This preserves an initial `SystemMessage` (if present) and ensures the first message after trimming is not a
        `ToolMessage` (some models/tooling can break when history starts with tool output).

        :param messages: message list to trim.
        :param max_tokens: token budget.
        :return: list[BaseMessage]: trimmed messages.
        """
        system_msg = None
        if messages and isinstance(messages[0], SystemMessage):
            system_msg = messages[0]
            messages_to_trim = messages[1:]
        else:
            messages_to_trim = messages

        trimmed_messages = trim_messages(
            messages_to_trim,
            token_counter=self.llm,
            max_tokens=max_tokens,
            strategy="last",
            include_system=True,
            allow_partial=False,
        )

        while trimmed_messages and isinstance(trimmed_messages[0], ToolMessage):
            trimmed_messages.pop(0)

        if system_msg:
            return [system_msg] + trimmed_messages
        return trimmed_messages

    @staticmethod
    def _compress(sessions: list[Session]) -> list[BaseMessage]:
        """
        Convert sessions into a single message history.

        Subclasses override this to apply different context compression strategies.

        :param sessions: past sessions.
        :return: list[BaseMessage]: flattened message history.
        """
        return DialogueBaseline._get_context(sessions)

    def process_dialogue(
            self,
            sessions: list[Session],
            system_prompt: str,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None,
    ) -> DialogueState:
        """
        Generate a response using the baseline approach (no explicit memory).

        :param sessions: past user/assistant/tool interactions.
        :param system_prompt: system prompt (used as the system message).
        :param structure: optional schema for structured output.
        :param tools: optional tools/functions specs for tool calling.
        :return: DialogueState: contains the prepared history and model response.
        """
        compressed_sessions: list[BaseMessage] = type(self)._compress(sessions)
        context: list[BaseMessage] = self._crop(compressed_sessions)

        system_instruction = self._prompt_builder.build(
            schema=structure,
            memory=MemorySections(),
            memory_mode="baseline",
        )

        chat_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("history")
        ])

        if structure or tools:
            chain = self._build_chain(chat_prompt, structure, tools)
        else:
            chain = chat_prompt | self.llm | StrOutputParser()

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(OutputParserException),
            reraise=True
        )
        def invoke_with_retry(input_data: dict[str, list[BaseMessage]]) -> Any:
            logging.info("Attempting to invoke chain...")
            return chain.invoke(input_data)

        # Ensure the unified SystemMessage is the first message in the flow.
        context_with_system: list[BaseMessage] = self._crop(
            [SystemMessage(content=system_instruction), *context]
        )

        with get_openai_callback() as cb:
            result = invoke_with_retry({"history": context_with_system})

            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens
            self.total_cost += cb.total_cost

        return DialogueState(
            dialogue_sessions=sessions,
            prepared_messages=context_with_system,
            query=system_prompt,
            _response=result,
            code_memory_storage=None,
            tool_memory_storage=None
        )
