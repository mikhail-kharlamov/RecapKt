import logging

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.algorithms.summarize_algorithms.core.models import ResponseContext, Session
from src.benchmark.tool_plan_benchmarking.tools_and_schemas.parsed_jsons import TOOLS
from src.utils.system_prompt_builder import MemorySections, SystemPromptBuilder


class ResponseGenerator:
    _MEMORY_MODE_BASELINE = "baseline"
    _MEMORY_MODE_MEMORY = "memory"
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
        structure: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        max_prompt_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self._structure = structure
        self._tools = tools
        self._max_prompt_tokens = max_prompt_tokens
        self._prompt_builder = SystemPromptBuilder()
        self._chain = self._build_chain()

    def _build_chain(self) -> Runnable:
        """Build the runnable used for response generation.

        `generate_response()` already assembles the full prompt as a list of messages
        (`full_history`), so the chain must accept a `list[BaseMessage]` directly.

        The branching logic for structured output and tool binding is intentionally kept.
        """
        if self._structure and not self._tools:
            return self._llm.with_structured_output(self._structure)

        if self._tools and not self._structure:
            return self._llm.bind_tools(self._tools)

        if self._tools and self._structure:
            tools = [self._get_return_action_plan(), *self._tools]
            return self._llm.bind_tools(tools)

        return self._llm | StrOutputParser()

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

    def _crop(self, messages: list[BaseMessage], max_tokens: int = 80000) -> list[BaseMessage]:
        """Trim a list of messages to fit into a token budget.

        Mirrors `DialogueBaseline._crop()` behavior:
        - preserves an initial `SystemMessage` (if present)
        - ensures the first message after trimming is not a `ToolMessage`
        """
        system_msg: SystemMessage | None = None
        if messages and isinstance(messages[0], SystemMessage):
            system_msg = messages[0]
            messages_to_trim = messages[1:]
        else:
            messages_to_trim = messages

        total_tokens_before_crop = self._llm.get_num_tokens_from_messages(messages_to_trim)
        logging.info(f"Total tokens before crop: {total_tokens_before_crop}")

        trimmed_messages = trim_messages(
            messages_to_trim,
            token_counter=self._llm,
            max_tokens=max_tokens,
            strategy="last",
            include_system=True,
            allow_partial=False,
        )

        while trimmed_messages and isinstance(trimmed_messages[0], ToolMessage):
            trimmed_messages.pop(0)

        if system_msg is not None:
            total_tokens_after_crop = self._llm.get_num_tokens_from_messages(trimmed_messages)
            logging.info(f"Total tokens after crop (without system message): {total_tokens_after_crop}")
            return [system_msg, *trimmed_messages]

        return trimmed_messages

    def _infer_memory_mode(self, memory: MemorySections) -> str:
        """Infer memory mode for system prompt rendering.

        We consider the run as "memory" if any memory section is present and non-empty.
        """
        sections = [
            memory.recap,
            memory.memory_bank,
            memory.code_knowledge,
            memory.tool_memory,
        ]
        has_memory = any((s or "").strip() != "" for s in sections)
        return self._MEMORY_MODE_MEMORY if has_memory else self._MEMORY_MODE_BASELINE

    def _build_system_message(
        self,
        *,
        memory: MemorySections,
    ) -> SystemMessage:
        """
        Build the single unified SystemMessage from Jinja2 templates.

        Order:
        1) introduction.j2
        2) memory injection (conditional)
        3) schema_and_tool.j2
        4) bridge_to_conversation.j2

        :param memory: memory sections to inject.
        :return: SystemMessage: unified system instruction.
        """
        system_prompt_text = self._prompt_builder.build(
            schema=self._structure,
            tools=TOOLS,
            memory=memory,
            memory_mode=self._infer_memory_mode(memory),
        )
        return SystemMessage(content=system_prompt_text)

    def generate_response(
        self,
        *,
        last_session: Session,
        user_query: str,
        memory: MemorySections,
    ) -> ResponseContext:
        """Generate a response using a single unified SystemMessage followed by the conversation history.

        :param last_session: the current conversation session (history used for response generation).
        :param user_query: latest user request (must become the last HumanMessage).
        :param memory: memory sections to inject into the system instruction.
        :return: ResponseContext: raw model output and the prepared history sent to the model.
        """
        try:
            history_messages: list[BaseMessage] = self._prepare_history(last_session, user_query)

            history_messages = [m for m in history_messages if not isinstance(m, SystemMessage)]

            system_message = self._build_system_message(memory=memory)

            final_prompt: list[BaseMessage] = [system_message, *history_messages]

            # Logging mirrors `DialogueBaseline`: show prompt token counts before/after crop.
            system_tokens = self._llm.get_num_tokens_from_messages([system_message])
            history_tokens = self._llm.get_num_tokens_from_messages(history_messages)
            total_tokens = self._llm.get_num_tokens_from_messages(final_prompt)
            logging.info(
                "Prompt tokens breakdown: system=%s history=%s total=%s",
                system_tokens,
                history_tokens,
                total_tokens,
            )

            prompt_to_invoke = final_prompt
            if self._max_prompt_tokens is not None:
                prompt_to_invoke = self._crop(final_prompt, max_tokens=self._max_prompt_tokens)
                logging.info(
                    "Prompt tokens after crop: total=%s",
                    self._llm.get_num_tokens_from_messages(prompt_to_invoke),
                )

            response = self._chain.invoke(prompt_to_invoke)

            return ResponseContext(response=response, prepared_history=prompt_to_invoke)

        except Exception as e:
            raise ConnectionError(f"API request failed: {str(e)}") from e
