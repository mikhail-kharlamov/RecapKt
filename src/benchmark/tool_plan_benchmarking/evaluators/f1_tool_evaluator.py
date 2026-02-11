import json

from decimal import Decimal
from typing import Any

from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    Session,
    ToolCallBlock,
)
from src.benchmark.models.dtos import MetricState
from src.benchmark.models.enums import MetricType
from src.benchmark.tool_plan_benchmarking.evaluators.base_evaluator import BaseEvaluator
from src.utils.semantic_similarity import SemanticSimilarity


class F1ToolEvaluator(BaseEvaluator):
    """
    Computes F1 between tools predicted by the model and tools used in a reference trace.

    The model is expected to return a structured response with a `plan_steps` list where tool calls are represented
    as entries with `kind == "tool_call"`.

    Modes:
    - default: compares only tool names
    - "strict": compares tool names + exact JSON arguments
    """

    def evaluate(
            self,
            sessions: list[Session],
            query: str,
            state: DialogueState,
            reference: list[BaseBlock] | None = None,
    ) -> MetricState:
        """
        Compute the F1 score for tool selection against a reference trace.

        :param sessions: previous sessions (unused here, but part of the evaluator interface).
        :param query: the evaluated user query.
        :param state: algorithm output state containing the model response.
        :param reference: reference blocks containing expected tool calls.
        :return: MetricState: metric name and computed value.
        """
        if reference is None:
            raise ValueError("Reference is required for F1 Tool evaluation.")

        if isinstance(state.response, str):
            raise ValueError("State response must be a structured object (dict), not a string.")

        reference_tools: set[str] = {
            tool.name
            for tool in reference
            if isinstance(tool, ToolCallBlock)
        }

        plan_steps = state.response.get("plan_steps", [])

        if self._mode == "strict":
            predicted_tools = F1ToolEvaluator._get_strict_matches(plan_steps, reference)
            metric_type = MetricType.F1_TOOL_STRICT
        elif self._mode == "arguments_similarity":
            predicted_tools = F1ToolEvaluator._get_strict_matches(plan_steps, reference)
            metric_type = MetricType.F1_TOOL_ARGUMENTS_SIMILARITY
        else:
            predicted_tools = F1ToolEvaluator._get_simple_matches(plan_steps)
            metric_type = MetricType.F1_TOOL

        true_positives = len(predicted_tools.intersection(reference_tools))
        false_positives = len(predicted_tools.difference(reference_tools))
        false_negatives = len(reference_tools.difference(predicted_tools))

        f1_score = F1ToolEvaluator._calculate_f1(true_positives, false_positives, false_negatives)

        return MetricState(
            metric_name=metric_type,
            metric_value=f1_score
        )

    @staticmethod
    def _get_simple_matches(plan_steps: list[dict[str, Any]]) -> set[str]:
        return {
            step.get("name", "")
            for step in plan_steps
            if step.get("kind") == "tool_call"
        }

    @staticmethod
    def _get_strict_matches(
            plan_steps: list[dict[str, Any]],
            reference: list[BaseBlock]
    ) -> set[str]:
        matches = set()

        ref_tool_blocks = [r for r in reference if isinstance(r, ToolCallBlock)]

        for step in plan_steps:
            if step.get("kind") != "tool_call":
                continue

            step_name = step.get("name", "")
            step_args = step.get("args", {})

            is_match = any(
                r.name.lower() == step_name and
                F1ToolEvaluator._compare_arguments(step_args, json.loads(r.arguments))
                for r in ref_tool_blocks
            )

            if is_match:
                matches.add(f"{step_name}|{step_args}")

        return matches

    @staticmethod
    def _get_args_similarity_matches(
            plan_steps: list[dict[str, Any]],
            reference: list[BaseBlock]
    ) -> set[str]:
        arguments_similarity_threshold: float = 0.7

        similarity = SemanticSimilarity() #TODO refactor it
        matches: set[str] = set()

        for step in plan_steps:
            if step.get("kind") != "tool_call":
                continue

            step_name = str(step.get("name", "")).strip()
            step_args = step.get("args", {})
            if not step_name:
                continue

            for block in reference:
                if isinstance(block, ToolCallBlock):
                    if block.name.lower() == step_name.lower():
                        score = similarity.compare_json(
                            json.loads(block.arguments),
                            step_args
                        )
                        if score >= arguments_similarity_threshold:
                            matches.add(f"{step_name}|{step_args}")

        return matches

    @staticmethod
    def _calculate_f1(tp: int, fp: int, fn: int) -> Decimal:
        """Compute F1 as an exact `Decimal`.

        Using Decimal avoids accumulating float rounding errors in downstream pipelines and is consistent with other
        benchmark DTOs that already allow `Decimal` metric values.
        """
        zero = Decimal("0")
        if tp == 0:
            return zero

        tp_d = Decimal(tp)
        precision = tp_d / Decimal(tp + fp)
        recall = tp_d / Decimal(tp + fn)

        if precision + recall == zero:
            return zero

        return Decimal(2) * (precision * recall) / (precision + recall)

    @staticmethod
    def _compare_arguments(args1: dict[str, Any], args2: dict[str, Any]) -> bool:
        return args1 == args2
