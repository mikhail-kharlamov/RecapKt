from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.benchmark.models.dtos import MetricState
from src.benchmark.models.enums import MetricType
from src.benchmark.tool_plan_benchmarking.evaluators.llm_as_a_judge_base_evaluator import LLMAsAJudgeBaseEvaluator


class DummyResult(BaseModel):
    score: int


class DummyJudgeEvaluator(LLMAsAJudgeBaseEvaluator):
    def _build_single_user_prompt(self, params: dict[str, Any]) -> str:
        return f"SINGLE: {params['x']}"

    def _build_pairwise_user_prompt(self, params: dict[str, Any]) -> str:
        return f"PAIRWISE: {params['x']}"

    def _get_single_result_model(self) -> type[BaseModel]:
        return DummyResult

    def _get_pairwise_result_model(self) -> type[BaseModel]:
        return DummyResult

    def evaluate(self, sessions, query, state, reference=None) -> MetricState:  # noqa: ANN001
        # Not needed for this unit test.
        return MetricState(metric_name=MetricType("COHERENCE"), metric_value=True)


def test_llm_as_a_judge_uses_system_prompt_builder_and_message_list() -> None:
    llm = MagicMock(spec=BaseChatModel)

    chain = MagicMock()
    chain.invoke.return_value = DummyResult(score=1)
    llm.with_structured_output.return_value = chain

    evaluator = DummyJudgeEvaluator(llm=llm)

    result = evaluator._invoke_single({"x": "hello"})
    assert isinstance(result, DummyResult)
    assert result.score == 1

    # Ensure we invoked the LLM with a list of messages.
    args, _kwargs = chain.invoke.call_args
    messages = args[0]

    assert isinstance(messages[0], SystemMessage)
    assert "Role" in messages[0].content  # comes from `introduction.j2`
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "SINGLE: hello"
