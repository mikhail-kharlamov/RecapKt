import seaborn as sns

from matplotlib import pyplot as plt

from src.benchmark.models.dtos import StatisticsDto
from src.benchmark.tool_plan_benchmarking.graphs.graph_builder import GraphBuilder


class GeneralTrends(GraphBuilder):
    """
    Line-plot of metric trend as the number of sessions grows.

    For each algorithm, computes the mean score per `sessions` bucket and draws a curve.
    """

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

        for algo, sub_df in summary.groupby("algorithm"):
            sub = sub_df.sort_values("sessions")

            x = sub["sessions"].to_numpy()
            y = sub["mean"].to_numpy().astype(float)

            ax.plot(x, y, marker="o", label=str(algo))

        ax.set_xlabel("Число сессий")
        ax.set_ylabel("F1_TOOL")
        ax.set_title("Тенденции качества по числу сессий " + title)
        ax.grid(True, which="major", axis="both", alpha=0.3)
        ax.legend(title="Алгоритм")

        fig.tight_layout()

        GeneralTrends._save_figure(path_to_save)
        plt.show()
