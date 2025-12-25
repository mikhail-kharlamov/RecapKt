import math
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.benchmarking.models.dtos import StatisticsDto
from src.benchmarking.models.enums import MetricType
from src.benchmarking.tool_metrics.graphs.graph_builder import GraphBuilder


class TrendsWithQuantiles(GraphBuilder):
    @staticmethod
    def build(statistics: StatisticsDto, path_to_save: str, title: str = "") -> None:
        sns.set_theme(style="whitegrid")
        df = TrendsWithQuantiles._runs_to_dataframe(statistics)
        summary = TrendsWithQuantiles.__summarize_for_bands(df)

        fig, ax = plt.subplots(figsize=(10, 6))

        for algo, sub in summary.groupby("algorithm"):
            sub = sub.sort_values("sessions")
            x = sub["sessions"].values
            y = sub["mean"].values
            y_low = sub["q_low"].values
            y_high = sub["q_high"].values

            ax.plot(x, y, marker="o", label=algo)
            ax.fill_between(x, y_low, y_high, alpha=0.2)

        ax.set_xlabel("Число сессий")
        ax.set_ylabel(df["metric"].iloc[0])
        ax.set_title("Тенденции качества " + title)
        ax.legend()
        fig.tight_layout()
        TrendsWithQuantiles._save_figure(path_to_save)
        plt.show()

    @staticmethod
    def __summarize_for_bands(df: pd.DataFrame, q_low=0.25, q_high=0.75) -> pd.DataFrame:
        grouped = df.groupby(["algorithm", "sessions"])["value"]
        summary = grouped.agg(
            mean="mean",
            q_low=lambda x: x.quantile(q_low),
            q_high=lambda x: x.quantile(q_high),
        ).reset_index()
        return summary

