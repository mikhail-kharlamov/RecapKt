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

    @staticmethod
    def _prepare_retrieval_information(
            code_memory: Session,
            tool_memory: Session,
            text_memory: Session
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content="Retrieval Information:")]
        if len(code_memory.messages) != 0:
            messages.append(SystemMessage(content="Code Memory:"))
            messages.extend(code_memory.to_langchain_messages())

        if len(tool_memory.messages) != 0:
            messages.append(SystemMessage(content="Tool Memory:"))
            messages.extend(tool_memory.to_langchain_messages())

        messages.append(SystemMessage(content="Text Memory:"))
        messages.extend(text_memory.to_langchain_messages())
        return messages

    def generate_response(
            self,
            last_session: Session,
            code_memory: Session,
            tool_memory: Session,
            text_memory: Session,
            query: str
    ) -> ResponseContext:
        """
        Generate a response using the configured LLM chain.

        The final prompt is built from:
        - a "Retrieval Information" section (code/tool/text memories)
        - the trimmed conversation history from `last_session`
        - the current `query` appended as the last user message

        :param last_session: the current conversation session (history used for response generation).
        :param code_memory: retrieved code memory blocks.
        :param tool_memory: retrieved tool memory blocks.
        :param text_memory: retrieved text memory blocks.
        :param query: the user query to answer.
        :return: ResponseContext: raw model output and the prepared history sent to the model.
        """
        try:
            memory_msg: list[BaseMessage] = ResponseGenerator._prepare_retrieval_information(
                code_memory,
                tool_memory,
                text_memory
            )

            history_messages: list[BaseMessage] = self._prepare_history(last_session, query)

            full_history: list[BaseMessage] = memory_msg + [SystemMessage(content="")] + history_messages

            response = self._chain.invoke({
                "history": full_history
            })

            return ResponseContext(response=response, prepared_history=full_history)

        except Exception as e:
            raise ConnectionError(f"API request failed: {str(e)}") from e
