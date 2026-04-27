import seaborn as sns

from matplotlib import pyplot as plt

from src.benchmark.tool_plan_benchmarking.graphs.graph_builder import GraphBuilder
from src.benchmark.tool_plan_benchmarking.statistics.dtos import StatisticsDto


class BoxPlot(GraphBuilder):
    """
    Box-plot visualization of metric distributions per algorithm.

    Useful for comparing variance and outliers across different dialogue systems/baselines.
    """

    @staticmethod
    def build(
        statistics: StatisticsDto,
        path_to_save: str,
        title: str = "",
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
        plt.close()
