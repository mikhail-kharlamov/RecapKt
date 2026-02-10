import json

from typing import Any, Callable

from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    Session,
    ToolCallBlock,
)
from src.benchmark.models.dtos import MetricState
from src.benchmark.models.enums import MetricType
from src.utils.semantic_similarity import SemanticSimilarity
from src.benchmark.tool_plan_benchmarking.evaluators.base_evaluator import BaseEvaluator


class F1ToolEvaluator(BaseEvaluator):
    """Compute F1 between predicted tool calls and reference tool calls.

    The model is expected to return a structured response with a `plan_steps` list where tool calls are represented
    as entries with `kind == "tool_call"`.

    Modes:
    - default/"simple"/any other: compare only tool names
    - "strict": compare tool name + exact JSON arguments
    - "arguments_similarity": compare tool name + semantic similarity of JSON arguments

    Note: the implementation intentionally evaluates *all predicted* tool calls; it does not pre-filter predictions by
    whether they match the reference.
    """

    _ARGUMENTS_SIMILARITY_THRESHOLD: float = 0.7

    def evaluate(
        self,
        sessions: list[Session],
        query: str,
        state: DialogueState,
        reference: list[BaseBlock] | None = None,
    ) -> MetricState:
        if reference is None:
            raise ValueError("Reference is required for F1 Tool evaluation.")

        if isinstance(state.response, str):
            raise ValueError("State response must be a structured object (dict), not a string.")

        ref_tool_blocks = [r for r in reference if isinstance(r, ToolCallBlock)]
        predicted_steps = [
            step
            for step in state.response.get("plan_steps", [])
            if isinstance(step, dict) and step.get("kind") == "tool_call"
        ]

        if self._mode == "strict":
            metric_type = MetricType.F1_TOOL_STRICT
            predicted_items = {self._canonical_tool_call(step.get("name", ""), step.get("args", {})) for step in predicted_steps}
            reference_items = {self._canonical_tool_call(r.name, json.loads(r.arguments)) for r in ref_tool_blocks}

        elif self._mode == "arguments_similarity":
            metric_type = MetricType.F1_TOOL_ARGUMENTS_SIMILARITY
            tp, fp, fn = self._calculate_similarity_counts(predicted_steps, ref_tool_blocks)
            return MetricState(metric_name=metric_type, metric_value=self._calculate_f1(tp, fp, fn))

        else:
            metric_type = MetricType.F1_TOOL
            predicted_items = {str(step.get("name", "")).lower() for step in predicted_steps}
            reference_items = {r.name.lower() for r in ref_tool_blocks}

        true_positives = len(predicted_items.intersection(reference_items))
        false_positives = len(predicted_items.difference(reference_items))
        false_negatives = len(reference_items.difference(predicted_items))

        return MetricState(
            metric_name=metric_type,
            metric_value=self._calculate_f1(true_positives, false_positives, false_negatives),
        )

    def _calculate_similarity_counts(
        self,
        predicted_steps: list[dict[str, Any]],
        reference_tools: list[ToolCallBlock],
    ) -> tuple[int, int, int]:
        """Compute TP/FP/FN using semantic similarity of arguments.

        A prediction is a TP if there exists an unmatched reference tool call with the same name and args similarity
        >= threshold. Remaining predictions are FP; remaining references are FN.
        """
        if self._similarity is None:
            self._similarity = SemanticSimilarity()

        matched_ref: set[int] = set()
        tp = 0
        fp = 0

        for step in predicted_steps:
            step_name = str(step.get("name", "")).lower()
            step_args = step.get("args", {})

            best_ref_idx: int | None = None
            best_score = 0.0

            for idx, ref in enumerate(reference_tools):
                if idx in matched_ref:
                    continue
                if ref.name.lower() != step_name:
                    continue

                try:
                    ref_args = json.loads(ref.arguments)
                except json.JSONDecodeError:
                    ref_args = {}

                score = self._similarity.compare_json(step_args, ref_args)
                if score > best_score:
                    best_score = score
                    best_ref_idx = idx

            if best_ref_idx is not None and best_score >= self._ARGUMENTS_SIMILARITY_THRESHOLD:
                matched_ref.add(best_ref_idx)
                tp += 1
            else:
                fp += 1

        fn = len(reference_tools) - len(matched_ref)
        return tp, fp, fn

    @staticmethod
    def _canonical_tool_call(name: str, args: Any) -> str:
        """Canonical string representation for strict comparisons."""
        try:
            args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except TypeError:
            args_str = json.dumps({}, sort_keys=True, ensure_ascii=False)
        return f"{name.lower()}|{args_str}"

    def _arguments_semantic_step_compare(self, step: dict[str, Any], ref_tool_block: ToolCallBlock) -> bool:
        step_name = step.get("name", "")
        step_args = step.get("args", {})

        if self._similarity is None:
            self._similarity = SemanticSimilarity()

        return ref_tool_block.name.lower() == step_name and self._similarity.compare_json(
            step_args,
            json.loads(ref_tool_block.arguments)
        ) >= self._ARGUMENTS_SIMILARITY_THRESHOLD

    @staticmethod
    def _simple_step_compare(step: dict[str, Any], ref_tool_block: ToolCallBlock) -> bool:
        step_name = step.get("name", "")
        return ref_tool_block.name.lower() == step_name

    @staticmethod
    def _strict_step_compare(step: dict[str, Any], ref_tool_block: ToolCallBlock) -> bool:
        step_name = step.get("name", "")
        step_args = step.get("args", {})

        return ref_tool_block.name.lower() == step_name and step_args == json.loads(ref_tool_block.arguments)

    @staticmethod
    def _calculate_f1(tp: int, fp: int, fn: int) -> float:
        if tp == 0:
            return 0.0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)
