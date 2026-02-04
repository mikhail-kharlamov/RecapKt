from __future__ import annotations

from src.benchmark.tool_plan_benchmarking.statistics.dtos import StatisticsDto


class StatisticsPrinter:
    """Human-friendly console output for `StatisticsDto`."""

    @staticmethod
    def print(stats: StatisticsDto) -> None:
        print("=== Statistics ===")
        for alg_stat in stats.algorithms:
            print(f"Algorithm: {alg_stat.name}")
            print(f"  Metric: {alg_stat.metric.value}")
            print(f"  Count of launches: {alg_stat.count_of_launches}")
            print(f"  Math. expectation: {alg_stat.mean:.4f}")
            print(f"  Variance: {alg_stat.variance:.4f}")
            print()
