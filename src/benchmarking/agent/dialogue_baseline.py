import logging
import os

from typing import Any

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOllama
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
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.benchmarking.baseline_logger import BaselineLogger
from src.benchmarking.prompts import BASELINE_PROMPT
from src.summarize_algorithms.core.dialogue import Dialogue
from src.summarize_algorithms.core.models import (
    DialogueState,
    OpenAIModels,
    Session, LocalModels,
)


class DialogueBaseline(Dialogue):
    def __init__(self, system_name: str, llm: BaseChatModel | None = None, is_local=False) -> None:
        load_dotenv()

        self.system_name = system_name

        self._initialize_model(llm, is_local)

        self.prompt_template = BASELINE_PROMPT
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0

        self.baseline_logger = BaselineLogger()

    def _initialize_model(self, llm: BaseChatModel | None = None, is_local: bool = False) -> None:
        if is_local:
            self.llm = ChatOllama(
                model=LocalModels.GEMMA_2_9_B.value,
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
            prompt: ChatPromptTemplate,  # <--- Добавили аргумент
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None
    ) -> Runnable:

        if structure and not tools:
            structured_llm = self.llm.with_structured_output(structure)
            return prompt | structured_llm  # <--- Используем переданный prompt

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
        return DialogueBaseline._get_context(sessions)

    def process_dialogue(
            self,
            sessions: list[Session],
            system_prompt: str,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None,
    ) -> DialogueState:
        compressed_sessions: list[BaseMessage] = type(self)._compress(sessions)
        context: list[BaseMessage] = self._crop(compressed_sessions)

        safe_system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", safe_system_prompt),
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

        with get_openai_callback() as cb:
            result = invoke_with_retry({"history": context})

            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens
            self.total_cost += cb.total_cost

        return DialogueState(
            dialogue_sessions=sessions,
            prepared_messages=context,
            query=system_prompt,
            _response=result,
            code_memory_storage=None,
            tool_memory_storage=None
        )
