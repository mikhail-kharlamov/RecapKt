import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.benchmarking.models.dtos import StatisticsDto
from src.benchmarking.models.enums import MetricType
from src.benchmarking.tool_metrics.graphs.graph_builder import GraphBuilder


class GeneralTrends(GraphBuilder):
    @staticmethod
    def build(
        statistics: StatisticsDto,
        path_to_save: str,
        title: str = "",
    ) -> None:
        sns.set_theme(style="whitegrid")

        df = GeneralTrends._runs_to_dataframe(statistics)

        grouped = df.groupby(["algorithm", "sessions"])["value"]
        summary = grouped.mean().reset_index(name="mean")

        fig, ax = plt.subplots(figsize=(10, 6))

        for algo, sub in summary.groupby("algorithm"):
            sub = sub.sort_values("sessions")
            x = sub["sessions"].values
            y = sub["mean"].values

            ax.plot(x, y, marker="o", label=algo)

        ax.set_xlabel("Число сессий")
        ax.set_ylabel("F1_TOOL")
        ax.set_title("Тенденции качества по числу сессий" + title)
        ax.grid(True, which="major", axis="both", alpha=0.3)
        ax.legend(title="Алгоритм")

        fig.tight_layout()

        GeneralTrends._save_figure(path_to_save)
        plt.show()
