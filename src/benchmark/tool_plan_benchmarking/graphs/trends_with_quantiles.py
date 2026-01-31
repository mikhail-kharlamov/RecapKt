import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from src.benchmark.models.dtos import StatisticsDto
from src.benchmark.tool_plan_benchmarking.graphs.graph_builder import GraphBuilder


class TrendsWithQuantiles(GraphBuilder):
    """
    Trend plot with uncertainty bands.

    Plots mean metric value over sessions and shades the inter-quantile band (default: 25th–75th percentile).
    """

    @staticmethod
    def build(statistics: StatisticsDto, path_to_save: str, title: str = "") -> None:
        sns.set_theme(style="whitegrid")
        df = TrendsWithQuantiles._runs_to_dataframe(statistics)
        summary = TrendsWithQuantiles.__summarize_for_bands(df)

        fig, ax = plt.subplots(figsize=(10, 6))

        for algo, sub_df in summary.groupby("algorithm"):
            sub = sub_df.sort_values("sessions")

            x = sub["sessions"].to_numpy()
            y = sub["mean"].to_numpy().astype(float)
            y_low = sub["q_low"].to_numpy().astype(float)
            y_high = sub["q_high"].to_numpy().astype(float)

            ax.plot(x, y, marker="o", label=str(algo))
            ax.fill_between(x, y_low, y_high, alpha=0.2)

        ax.set_xlabel("Число сессий")

        metric_label = str(df["metric"].iloc[0]) if "metric" in df.columns and not df.empty else "Metric"
        ax.set_ylabel(metric_label)

        ax.set_title("Тенденции качества " + title)
        ax.legend()
        fig.tight_layout()
        TrendsWithQuantiles._save_figure(path_to_save)
        plt.show()

    @staticmethod
    def __summarize_for_bands(df: pd.DataFrame, q_low: float = 0.25, q_high: float = 0.75) -> pd.DataFrame:
        grouped = df.groupby(["algorithm", "sessions"])["value"]
        summary = grouped.agg(
            mean="mean",
            q_low=lambda x: x.quantile(q_low),
            q_high=lambda x: x.quantile(q_high),
        ).reset_index()
        return summary
