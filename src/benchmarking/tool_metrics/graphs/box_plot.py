from matplotlib import pyplot as plt

from src.benchmarking.models.dtos import StatisticsDto
from src.benchmarking.models.enums import MetricType
from src.benchmarking.tool_metrics.graphs.graph_builder import GraphBuilder

import seaborn as sns


class BoxPlot(GraphBuilder):
    @staticmethod
    def build(
            statistics: StatisticsDto,
            path_to_save: str,
            title: str = ""
    ) -> None:
        df = BoxPlot._runs_to_dataframe(statistics)

        sns.set_theme(style="whitegrid")

        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=df,
            x="algorithm",
            y="value",
            width=0.6,
            showfliers=True,
        )

        sns.stripplot(
            data=df,
            x="algorithm",
            y="value",
            color="black",
            size=3,
            alpha=0.6,
            jitter=0.15,
            ax=ax,
        )

        ax.set_title("Распределение значений " + title)
        ax.set_xlabel("Алгоритм")
        ax.set_ylabel(df["metric"].iloc[0])  # имя метрики из dto

        plt.xticks(rotation=20)
        plt.tight_layout()
        BoxPlot._save_figure(path_to_save)
        plt.show()

