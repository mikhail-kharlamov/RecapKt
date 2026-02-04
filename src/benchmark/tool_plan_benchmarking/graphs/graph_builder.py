from abc import ABC, abstractmethod

import pandas as pd

from matplotlib import pyplot as plt

from src.benchmark.tool_plan_benchmarking.statistics.dtos import (
    AlgorithmStatistics,
    StatisticsDto,
)


class GraphBuilder(ABC):
    """
    Base class for graph/plot builders used in tool-metrics benchmark.

    Implementations take a `StatisticsDto` (a list of runs) and render a figure to disk. Helpers in this base class
    convert runs into a Pandas `DataFrame` and save the active Matplotlib figure.
    """

    @staticmethod
    @abstractmethod
    def build(statistics: StatisticsDto, path_to_save: str, title: str = "") -> None:
        """
        Render a graph for the provided statistics and save it to `path_to_save`.

        :param statistics: aggregated run statistics.
        :param path_to_save: output image path.
        :param title: optional title suffix.
        :return: None
        """
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

        # Ensure expected columns exist even when there are no rows.
        columns = ["algorithm", "metric", "sessions", "value"]
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _save_figure(path: str = "graph.png") -> None:
        # Some callers pass an empty string; treat it as "use default".
        safe_path = path or "graph.png"
        plt.savefig(safe_path, dpi=300, bbox_inches="tight")
