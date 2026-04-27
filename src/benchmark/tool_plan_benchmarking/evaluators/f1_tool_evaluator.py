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
            raise ValueError(
                "State response must be a structured object (dict), not a string."
            )

        plan_steps = state.response.get("plan_steps", [])

        if self._mode == "strict":
            predicted_tools = F1ToolEvaluator._get_strict_matches(plan_steps, reference)
            metric_type = MetricType.F1_TOOL_STRICT
            strict_reference_tools: set[str] = {
                f"{tool.name}|{tool.arguments}"
                for tool in reference
                if isinstance(tool, ToolCallBlock)
            }
            f1_score = F1ToolEvaluator._calculate_f1(
                strict_reference_tools, predicted_tools
            )
        elif self._mode == "arguments_similarity":
            metric_type = MetricType.F1_TOOL_ARGUMENTS_SIMILARITY
            f1_score = F1ToolEvaluator._calculate_args_similarity_f1(
                plan_steps, reference
            )
        else:
            predicted_tools = F1ToolEvaluator._get_simple_matches(plan_steps)
            metric_type = MetricType.F1_TOOL
            simple_reference_tools: set[str] = {
                tool.name for tool in reference if isinstance(tool, ToolCallBlock)
            }
            f1_score = F1ToolEvaluator._calculate_f1(
                simple_reference_tools, predicted_tools
            )

        return MetricState(metric_name=metric_type, metric_value=f1_score)

    @staticmethod
    def _get_simple_matches(plan_steps: list[dict[str, Any]]) -> set[str]:
        return {
            step.get("name", "")
            for step in plan_steps
            if step.get("kind") == "tool_call"
        }

    @staticmethod
    def _get_strict_matches(
        plan_steps: list[dict[str, Any]], reference: list[BaseBlock]
    ) -> set[str]:
        matches = set()

        ref_tool_blocks = [r for r in reference if isinstance(r, ToolCallBlock)]

        for step in plan_steps:
            if step.get("kind") != "tool_call":
                continue

            step_name = step.get("name", "")
            step_args = step.get("args", {})

            is_match = any(
                r.name.lower() == step_name
                and F1ToolEvaluator._compare_arguments(
                    step_args, json.loads(r.arguments)
                )
                for r in ref_tool_blocks
            )

            if is_match:
                matches.add(f"{step_name}|{step_args}")

        return matches

    @staticmethod
    def _calculate_args_similarity_f1(
        plan_steps: list[dict[str, Any]], reference: list[BaseBlock]
    ) -> Decimal:
        arguments_similarity_threshold: float = 0.5

        similarity = SemanticSimilarity()
        reference_calls: list[tuple[str, dict[str, Any]]] = []
        predicted_calls: list[tuple[str, dict[str, Any]]] = []

        for block in reference:
            if not isinstance(block, ToolCallBlock):
                continue
            reference_calls.append(
                (block.name.lower(), F1ToolEvaluator._safe_load_dict(block.arguments))
            )

        for step in plan_steps:
            if step.get("kind") != "tool_call":
                continue
            step_name = str(step.get("name", "")).strip().lower()
            step_args = step.get("args", {})
            if not step_name:
                continue
            predicted_calls.append(
                (step_name, step_args if isinstance(step_args, dict) else {})
            )

        matched_reference_indices: set[int] = set()
        tp = 0
        fp = 0

        for predicted_name, predicted_args in predicted_calls:
            best_idx, best_score = F1ToolEvaluator._find_best_reference_match(
                predicted_name=predicted_name,
                predicted_args=predicted_args,
                reference_calls=reference_calls,
                matched_reference_indices=matched_reference_indices,
                similarity=similarity,
            )

            if best_idx is not None and best_score >= arguments_similarity_threshold:
                tp += 1
                matched_reference_indices.add(best_idx)
            else:
                fp += 1

        fn = len(reference_calls) - len(matched_reference_indices)
        return F1ToolEvaluator._calculate_f1_from_counts(tp=tp, fp=fp, fn=fn)

    @staticmethod
    def _calculate_f1(reference_tools: set[str], predicted_tools: set[str]) -> Decimal:
        """Compute F1 as an exact `Decimal`.

        Using Decimal avoids accumulating float rounding errors in downstream pipelines and is consistent with other
        benchmark DTOs that already allow `Decimal` metric values.
        """
        tp = len(predicted_tools.intersection(reference_tools))
        fp = len(predicted_tools.difference(reference_tools))
        fn = len(reference_tools.difference(predicted_tools))

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
    def _calculate_f1_from_counts(tp: int, fp: int, fn: int) -> Decimal:
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
    def _safe_load_dict(raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _find_best_reference_match(
        predicted_name: str,
        predicted_args: dict[str, Any],
        reference_calls: list[tuple[str, dict[str, Any]]],
        matched_reference_indices: set[int],
        similarity: SemanticSimilarity,
    ) -> tuple[int | None, float]:
        best_idx: int | None = None
        best_score = -1.0

        for idx, (ref_name, ref_args) in enumerate(reference_calls):
            if idx in matched_reference_indices or ref_name != predicted_name:
                continue

            score = similarity.compare_json(ref_args, predicted_args)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx, best_score

    @staticmethod
    def _compare_arguments(args1: dict[str, Any], args2: dict[str, Any]) -> bool:
        return args1 == args2
