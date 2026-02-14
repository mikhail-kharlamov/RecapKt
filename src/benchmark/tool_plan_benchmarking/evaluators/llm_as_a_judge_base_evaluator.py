from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from src.algorithms.summarize_algorithms.core.models import BaseBlock, DialogueState, OpenAIModels, Session
from src.benchmark.models.dtos import MetricState
from src.benchmark.tool_plan_benchmarking.evaluators.base_evaluator import BaseEvaluator
from src.utils.system_prompt_builder import MemorySections, SystemPromptBuilder


class LLMAsAJudgeBaseEvaluator(BaseEvaluator):
    """Base class for evaluators that delegate metric computation to an LLM "judge"."""

    def __init__(
        self,
        mode: str | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        super().__init__(mode=mode)

        load_dotenv()

        # Allow passing a fake/mock LLM without requiring OPENAI_API_KEY.
        if llm is not None:
            self.llm = llm
        else:
            api_key: str | None = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable is not loaded")

            self.llm = ChatOpenAI(
                model=OpenAIModels.GPT_4_O_MINI.value,
                api_key=SecretStr(api_key),
            )

        self._prompt_builder = SystemPromptBuilder()
        self._system_message = SystemMessage(content=self._build_system_prompt())

    def _build_system_prompt(self) -> str:
        """Build a unified system prompt for the judge."""
        return self._prompt_builder.build(
            schema=None,
            tools=None,
            memory=MemorySections(),
            memory_mode="baseline",
            examples=self._get_judge_examples(),
        )

    def _get_judge_examples(self) -> str:
        return ""

    @abstractmethod
    def _build_single_user_prompt(self, params: dict[str, Any]) -> str:
        """Render the HumanMessage content for a single-option evaluation."""

    @abstractmethod
    def _build_pairwise_user_prompt(self, params: dict[str, Any]) -> str:
        """Render the HumanMessage content for a pairwise evaluation."""

    @abstractmethod
    def _get_single_result_model(self) -> type[BaseModel]:
        """Structured output model for single-option evaluation."""

    @abstractmethod
    def _get_pairwise_result_model(self) -> type[BaseModel]:
        """Structured output model for pairwise evaluation."""

    def _build_messages(self, user_prompt: str) -> list[BaseMessage]:
        return [self._system_message, HumanMessage(content=user_prompt)]

    def _invoke_single(self, params: dict[str, Any]) -> BaseModel:
        model = self._get_single_result_model()
        chain = self.llm.with_structured_output(model)
        return self._safe_invoke(chain, self._build_messages(self._build_single_user_prompt(params)))

    def _invoke_pairwise(self, params: dict[str, Any]) -> BaseModel:
        model = self._get_pairwise_result_model()
        chain = self.llm.with_structured_output(model)
        return self._safe_invoke(chain, self._build_messages(self._build_pairwise_user_prompt(params)))

    @staticmethod
    def _safe_invoke(chain: Any, messages: list[BaseMessage]) -> Any:
        try:
            return chain.invoke(messages)
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e

    @abstractmethod
    def evaluate(
        self,
        sessions: list[Session],
        query: str,
        state: DialogueState,
        reference: list[BaseBlock] | None = None,
    ) -> MetricState:
        ...
