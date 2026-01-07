from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.summarize_algorithms.core.models import ResponseContext, Session


class ResponseGenerator:
    def __init__(self,
                 llm: BaseChatModel,
                 prompt_template: ChatPromptTemplate,
                 structure: dict[str, Any] | None = None,
                 tools: list[dict[str, Any]] | None = None
                 ) -> None:
        self._llm = llm
        self._prompt_template = prompt_template
        self._structure = structure
        self._tools = tools
        self._chain = self._build_chain()

    def _build_chain(self) -> Runnable:
        if self._structure and not self._tools:
            structured_llm = self._llm.with_structured_output(self._structure)
            return self._prompt_template | structured_llm

        if self._tools and not self._structure:
            llm_with_tools = self._llm.bind_tools(self._tools)
            return self._prompt_template | llm_with_tools

        if self._tools and self._structure:
            tools = [self._get_return_action_plan(), *self._tools]
            llm_with_tools = self._llm.bind_tools(tools)
            return self._prompt_template | llm_with_tools

        return self._prompt_template | self._llm | StrOutputParser()

    def _get_return_action_plan(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "return_action_plan",
                "description": "Return the final JSON action plan strictly matching the schema.",
                "parameters": self._structure,
            },
        }

    def _prepare_history(self, sessions: Session, query: str) -> list[BaseMessage]:
        trimmed_history = trim_messages(
            sessions.to_langchain_messages(),
            token_counter=self._llm,
            max_tokens=100000,
            strategy="last",
            include_system=False,
            allow_partial=False,
        )

        trimmed_history.append(HumanMessage(content=query))

        return trimmed_history

    def generate_response(
            self,
            last_session: Session,
            code_memory: Session,
            tool_memory: Session,
            text_memory: Session,
            query: str
    ) -> ResponseContext:
        try:
            memory_context = f"""
            Retrieval Information:
            - Code Memory: {code_memory}
            - Tool Memory: {tool_memory}
            - Text Memory: {text_memory}
            """

            memory_msg = SystemMessage(content=memory_context)

            history_messages = self._prepare_history(last_session, query)

            full_history = [memory_msg] + history_messages

            response = self._chain.invoke({
                "history": full_history
            })

            return ResponseContext(response=response, prepared_history=full_history)

        except Exception as e:
            raise ConnectionError(f"API request failed: {str(e)}") from e
