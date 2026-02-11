from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Generic, TypeVar

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    OpenAIModels,
    Session,
)
from src.benchmark.models.dtos import MetricState
from src.benchmark.tool_plan_benchmarking.evaluators.base_evaluator import BaseEvaluator
from src.utils.system_prompt_builder import MemorySections, SystemPromptBuilder


SingleResultType = TypeVar("SingleResultType", bound=BaseModel)
PairwiseResultType = TypeVar("PairwiseResultType", bound=BaseModel)


class LLMAsAJudgeBaseEvaluator[SingleResultType, PairwiseResultType](BaseEvaluator):
    """Base class for evaluators that delegate metric computation to an LLM "judge"."""

    _MEMORY_MODE: str = "baseline"

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
        """Build a unified system prompt for the judge.

        We intentionally don't pass any tools/schema here: the judge is expected to respond with structured output
        enforced by `with_structured_output(...)`, and it should not call tools.

        Subclasses may inject additional judging instructions via `_get_judge_examples()`.
        """
        return self._prompt_builder.build(
            schema=None,
            tools=None,
            memory=MemorySections(),
            memory_mode=self._MEMORY_MODE,
            examples=self._get_judge_examples(),
        )

    def _get_judge_examples(self) -> str:
        """Optional extra system-level instructions/examples for the judge."""
        return ""

    @abstractmethod
    def _build_single_user_prompt(self, params: dict[str, Any]) -> str:
        """Render the HumanMessage content for a single-option evaluation."""

    @abstractmethod
    def _build_pairwise_user_prompt(self, params: dict[str, Any]) -> str:
        """Render the HumanMessage content for a pairwise evaluation."""

    @abstractmethod
    def _get_single_result_model(self) -> type[SingleResultType]:
        """Pydantic model for the single-option judging result."""

    @abstractmethod
    def _get_pairwise_result_model(self) -> type[PairwiseResultType]:
        """Pydantic model for the pairwise judging result."""

    def _invoke_single(self, params: dict[str, Any]) -> SingleResultType:
        chain = self.llm.with_structured_output(self._get_single_result_model())
        messages: list[BaseMessage] = [
            self._system_message,
            HumanMessage(content=self._build_single_user_prompt(params)),
        ]
        return self._safe_invoke(chain, messages)

    def _invoke_pairwise(self, params: dict[str, Any]) -> PairwiseResultType:
        chain = self.llm.with_structured_output(self._get_pairwise_result_model())
        messages: list[BaseMessage] = [
            self._system_message,
            HumanMessage(content=self._build_pairwise_user_prompt(params)),
        ]
        return self._safe_invoke(chain, messages)

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
