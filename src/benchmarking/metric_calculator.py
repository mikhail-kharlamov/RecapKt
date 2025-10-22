import abc
import json

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pydantic import BaseModel

from src.benchmarking.deserialize_mcp_data import MCPDataset
from src.benchmarking.llm_evaluation import ComparisonResult
from src.summarize_algorithms.recsum.dialogue_system import RecsumDialogueSystem


@dataclass
class RawSemanticData:
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    f1: List[float] = field(default_factory=list)


@dataclass
class RawLLMData:
    faithfulness: List[float] = field(default_factory=list)
    informativeness: List[float] = field(default_factory=list)
    coherency: List[float] = field(default_factory=list)


@dataclass
class MetricStats:
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricStats":
        if not values:
            return cls()

        np_values = np.array(values)
        return cls(
            mean=float(np.mean(np_values)),
            std=float(np.std(np_values)),
            min=float(np.min(np_values)),
            max=float(np.max(np_values)),
            count=len(values),
        )


@dataclass
class SystemResults:
    semantic_precision: MetricStats = field(default_factory=MetricStats)
    semantic_recall: MetricStats = field(default_factory=MetricStats)
    semantic_f1: MetricStats = field(default_factory=MetricStats)

    llm_faithfulness: MetricStats = field(default_factory=MetricStats)
    llm_informativeness: MetricStats = field(default_factory=MetricStats)
    llm_coherency: MetricStats = field(default_factory=MetricStats)


@dataclass
class PairwiseResults:
    faithfulness: Dict[str, int] = field(
        default_factory=lambda: {"recsum": 0, "baseline": 0, "draw": 0}
    )
    informativeness: Dict[str, int] = field(
        default_factory=lambda: {"recsum": 0, "baseline": 0, "draw": 0}
    )
    coherency: Dict[str, int] = field(
        default_factory=lambda: {"recsum": 0, "baseline": 0, "draw": 0}
    )

    def get_total_count(self) -> int:
        return sum(self.faithfulness.values())


@dataclass
class MCPResult:
    metadata: Dict[str, Any] = field(default_factory=dict)
    recsum_results: SystemResults = field(default_factory=SystemResults)
    pairwise_results: PairwiseResults = field(default_factory=PairwiseResults)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MCPResponseResults(MCPResult):
    baseline_results: SystemResults = field(default_factory=SystemResults)


@dataclass
class MCPMemoryResults(MCPResult):
    memory_bank_results: SystemResults = field(default_factory=SystemResults)


class CalculateMCPMetrics(abc.ABC):
    def __init__(self, n_samples: int = 30):
        self.dataset = MCPDataset(n_samples)
        self.recsum = RecsumDialogueSystem()

        self._recsum_semantic_data = RawSemanticData()
        self._recsum_llm_data = RawLLMData()
        self._pairwise_data = PairwiseResults()

        self.n_samples = n_samples

        self._is_calculated = False

    @property
    @abc.abstractmethod
    def results(self) -> MCPResponseResults:
        pass

    @abc.abstractmethod
    def calculate(self) -> None:
        pass

    def _update_pairwise_counts(self, score: BaseModel, recsum_first: bool) -> None:
        metrics = ["faithfulness", "informativeness", "coherency"]

        for metric in metrics:
            score_value = getattr(score, metric)
            result_dict = getattr(self._pairwise_data, metric)

            if score_value == ComparisonResult.OPTION_1_BETTER:
                winner = "recsum" if recsum_first else "baseline"
                result_dict[winner] += 1
            elif score_value == ComparisonResult.OPTION_2_BETTER:
                winner = "baseline" if recsum_first else "recsum"
                result_dict[winner] += 1
            else:
                result_dict["draw"] += 1

    def save_results_to_json(self, filepath: str = None) -> str:
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"mcp_results_{timestamp}.json"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        results_dict = self.results.to_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        return filepath

    def _print_semantic_results(self, results: MCPResponseResults) -> None:
        print("=" * 50)
        print("SEMANTIC EVALUATION RESULTS")
        print("=" * 50)
        print(
            f"RecSum    - Precision: {results.recsum_results.semantic_precision.mean:.4f}"
            f" (±{results.recsum_results.semantic_precision.std:.4f}), "
            f"Recall: {results.recsum_results.semantic_recall.mean:.4f}"
            f" (±{results.recsum_results.semantic_recall.std:.4f}), "
            f"F1: {results.recsum_results.semantic_f1.mean:.4f}"
            f" (±{results.recsum_results.semantic_f1.std:.4f})"
        )
        print(
            f"Baseline  - Precision: {results.baseline_results.semantic_precision.mean:.4f}"
            f" (±{results.baseline_results.semantic_precision.std:.4f}), "
            f"Recall: {results.baseline_results.semantic_recall.mean:.4f}"
            f" (±{results.baseline_results.semantic_recall.std:.4f}), "
            f"F1: {results.baseline_results.semantic_f1.mean:.4f}"
            f" (±{results.baseline_results.semantic_f1.std:.4f})"
        )
        print()

    def _print_llm_single_results(self, results: MCPResponseResults) -> None:
        print("=" * 50)
        print("LLM SINGLE EVALUATION RESULTS")
        print("=" * 50)
        print(
            f"RecSum    - Faithfulness: {results.recsum_results.llm_faithfulness.mean:.2f}"
            f" (±{results.recsum_results.llm_faithfulness.std:.2f}), "
            f"Informativeness: {results.recsum_results.llm_informativeness.mean:.2f}"
            f" (±{results.recsum_results.llm_informativeness.std:.2f}), "
            f"Coherency: {results.recsum_results.llm_coherency.mean:.2f}"
            f" (±{results.recsum_results.llm_coherency.std:.2f})"
        )
        print(
            f"Baseline  - Faithfulness: {results.baseline_results.llm_faithfulness.mean:.2f}"
            f" (±{results.baseline_results.llm_faithfulness.std:.2f}), "
            f"Informativeness: {results.baseline_results.llm_informativeness.mean:.2f}"
            f" (±{results.baseline_results.llm_informativeness.std:.2f}), "
            f"Coherency: {results.baseline_results.llm_coherency.mean:.2f}"
            f" (±{results.baseline_results.llm_coherency.std:.2f})"
        )
        print()

    def _print_llm_pairwise_results(self, results: MCPResponseResults) -> None:
        total_count = results.pairwise_results.get_total_count()

        if total_count == 0:
            print("No pairwise evaluations completed.")
            return

        print("=" * 50)
        print("LLM PAIRWISE EVALUATION RESULTS")
        print("=" * 50)

        metrics = ["faithfulness", "informativeness", "coherency"]
        for metric in metrics:
            result_dict = getattr(results.pairwise_results, metric)
            recsum_wins = result_dict["recsum"]
            baseline_wins = result_dict["baseline"]
            draws = result_dict["draw"]

            print(
                f"{metric.capitalize():<15}: RecSum {recsum_wins}/{total_count}"
                f" ({recsum_wins / total_count * 100:.1f}%), "
                f"Baseline {baseline_wins}/{total_count} ({baseline_wins / total_count * 100:.1f}%), "
                f"Draws {draws}/{total_count} ({draws / total_count * 100:.1f}%)"
            )
