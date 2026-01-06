from abc import ABC, abstractmethod

import pandas as pd

from matplotlib import pyplot as plt

from src.benchmarking.models.dtos import AlgorithmStatistics, StatisticsDto


class GraphBuilder(ABC):
    @staticmethod
    @abstractmethod
    def build(statistics: StatisticsDto, path_to_save: str, title: str = "") -> None:
        ...

    @staticmethod
    def _runs_to_dataframe(stats: StatisticsDto) -> pd.DataFrame:
        algorithms: list[AlgorithmStatistics] = stats.algorithms
        rows = [
            {
                "algorithm": alg.name,
                "metric": alg.metric.value,
                "sessions": run.sessions,
                "value": run.value,
            }
            for alg in algorithms
            for run in alg.runs
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _save_figure(path: str = "graph.png") -> None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
