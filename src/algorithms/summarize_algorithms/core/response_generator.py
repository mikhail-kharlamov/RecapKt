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

from src.algorithms.summarize_algorithms.core.models import ResponseContext, Session
from src.utils.system_prompt_builder import MemorySections, SystemPromptBuilder


class ResponseGenerator:
    """
    Generates the final assistant response given:
    - the last dialogue session
    - retrieved memory (code/tool/text)
    - the current user query

    Depending on configuration, it can:
    - return plain text (`StrOutputParser`)
    - call tools (`bind_tools`)
    - return structured JSON (`with_structured_output`)
    """

    def __init__(
            self,
            llm: BaseChatModel,
            prompt_template: ChatPromptTemplate,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self._llm = llm
        self._prompt_template = prompt_template
        self._structure = structure
        self._tools = tools
        self._prompt_builder = SystemPromptBuilder()
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
        """
        Build an auxiliary tool spec that forces the model to return the structured JSON according to `self._structure`.

        :return: dict[str, Any]: tool spec for `bind_tools`.
        """
        return {
            "type": "function",
            "function": {
                "name": "return_action_plan",
                "description": "Return the final JSON action plan strictly matching the schema.",
                "parameters": self._structure,
            },
        }

    def _prepare_history(self, sessions: Session, user_query: str) -> list[BaseMessage]:
        """
        Prepare the message history that will follow the unified system instruction.

        Ensures the last message in the returned list is the latest user request.

        :param sessions: conversation session used as history.
        :param user_query: latest user request (may already be present as the last user message in `sessions`).
        :return: list[BaseMessage]: prepared history messages (no SystemMessage).
        """
        trimmed_history = trim_messages(
            sessions.to_langchain_messages(),
            token_counter=self._llm,
            max_tokens=100000,
            strategy="last",
            include_system=False,
            allow_partial=False,
        )

        if user_query.strip() != "":
            should_append = True
            if trimmed_history and isinstance(trimmed_history[-1], HumanMessage):
                should_append = trimmed_history[-1].content != user_query
            if should_append:
                trimmed_history.append(HumanMessage(content=user_query))

        return trimmed_history

    def _build_unified_system_message(
            self,
            *,
            memory: MemorySections,
            memory_mode: str,
    ) -> SystemMessage:
        """
        Build the single unified SystemMessage from Jinja2 templates.

        Order:
        1) introduction.j2
        2) memory injection (conditional)
        3) schema_and_tool.j2
        4) bridge_to_conversation.j2

        :param memory: memory sections to inject.
        :param memory_mode: "baseline" or "memory".
        :return: SystemMessage: unified system instruction.
        """
        system_prompt_text = self._prompt_builder.build(
            schema=self._structure,
            memory=memory,
            memory_mode=memory_mode,
        )
        return SystemMessage(content=system_prompt_text)

    def generate_response(
            self,
            *,
            last_session: Session,
            user_query: str,
            memory: MemorySections,
            memory_mode: str,
    ) -> ResponseContext:
        """
        Generate a response using a single unified SystemMessage followed by the conversation history.

        :param last_session: the current conversation session (history used for response generation).
        :param user_query: latest user request (must become the last HumanMessage).
        :param memory: memory sections to inject into the system instruction.
        :param memory_mode: "baseline" or "memory".
        :return: ResponseContext: raw model output and the prepared history sent to the model.
        """
        try:
            history_messages: list[BaseMessage] = self._prepare_history(last_session, user_query)

            history_messages = [m for m in history_messages if not isinstance(m, SystemMessage)]

            system_message = self._build_unified_system_message(memory=memory, memory_mode=memory_mode)

            full_history: list[BaseMessage] = [system_message, *history_messages]

            response = self._chain.invoke({
                "history": full_history
            })

            return ResponseContext(response=response, prepared_history=full_history)

        except Exception as e:
            raise ConnectionError(f"API request failed: {str(e)}") from e
