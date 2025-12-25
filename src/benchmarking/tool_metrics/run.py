import json
import logging
from pathlib import Path
from typing import Any

import tiktoken
from jinja2 import Environment, FileSystemLoader

from src.benchmarking.agent.baseline import DialogueBaseline
from src.benchmarking.baseline_logger import BaselineLogger
from src.benchmarking.memory_logger import MemoryLogger
from src.benchmarking.models.enums import MetricType
from src.benchmarking.tool_metrics.evaluators.f1_tool_evaluator import F1ToolEvaluator
from src.benchmarking.tool_metrics.graphs.general_trends import GeneralTrends
from src.benchmarking.tool_metrics.graphs.graph_builder import GraphBuilder
from src.benchmarking.tool_metrics.load_session import Loader
from src.benchmarking.models.dtos import QueryAndReference, StatisticsDto, BaseRecord, MemoryRecord, TokenInfo, \
    MODEL_PRICES, AlgorithmStatistics
from src.benchmarking.tool_metrics.statistics_calc import Statistics
from src.summarize_algorithms.core.models import BaseBlock, Session, OpenAIModels
from src.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueSystem,
)
from src.summarize_algorithms.recsum.dialogue_system import RecsumDialogueSystem
from src.utils.configure_logs import configure_logs


class Runner:
    def __init__(self, templates_dir: str = "prompts") -> None:
        self.logger = logging.getLogger()
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=True,
            trim_blocks=True
        )

    def run(self, name: str) -> None:
        memory_logger = MemoryLogger()
        baseline_logger = BaselineLogger()

        if name == "base_recsum":
            algorithm = RecsumDialogueSystem(embed_code=False, embed_tool=False, system_name="BaseRecsum")
        elif name == "base_memory_bank":
            algorithm = MemoryBankDialogueSystem(embed_code=False, embed_tool=False, system_name="BaseMemoryBank")
        elif name == "rag_recsum":
            algorithm = RecsumDialogueSystem(embed_code=True, embed_tool=True, system_name="RagRecsum")
        elif name == "rag_memory_bank":
            algorithm = MemoryBankDialogueSystem(embed_code=True, embed_tool=True, system_name="RagMemoryBank")
        elif name == "full_baseline":
            algorithm = DialogueBaseline("FullBaseline")
        else:
            algorithm = DialogueBaseline("LastBaseline")

        self.logger.info("Start parsing session")
        past_interactions: list[Session] = []

        json_file_template: str = "*.json"
        path_data_type_1: Path = Path("/Users/mikhailkharlamov/Documents/.../NewDataSet/data_type_1")
        for file in path_data_type_1.glob(json_file_template):
            past_interactions.append(
                Loader.load_session_data_type_1(file)
            )

        path_data_type_2: Path = Path("/Users/mikhailkharlamov/Documents/.../NewDataSet/data_type_2")
        for file in path_data_type_2.glob(json_file_template):
            past_interactions.append(
                Loader.load_session_data_type_2(file)
            )

        gold_session: Session = Loader.load_session_data_type_1(
            "/Users/mikhailkharlamov/Documents/.../NewDataSet/gold_session.json"
        )

        query_and_reference = self.__execute_query_and_reference(gold_session)
        query, reference = query_and_reference.query, query_and_reference.reference
        prompt = self.__prepare_query_for_the_first_stage(query)

        f1_tool_evaluator_strict = F1ToolEvaluator("st")
        f1_tool_evaluator = F1ToolEvaluator()

        #1, 3, 5, 7, 9, 11,
        for count_of_sessions in [1, 3, 5, 7, 9, 11, 13, 15]:
            subdirectory: Path = Path(str(count_of_sessions))

            if name == "full_baseline":
                self.logger.info("Start evaluating full baseline statistics")
                statistics: StatisticsDto = Statistics.calculate(
                    5,
                    [algorithm],
                    [f1_tool_evaluator, f1_tool_evaluator_strict],
                    past_interactions,
                    count_of_sessions,
                    gold_session,
                    prompt,
                    reference,
                    baseline_logger,
                    None,
                    subdirectory,
                    True
                )
            elif name == "last_baseline":
                self.logger.info("Start evaluating last baseline statistics")
                statistics: StatisticsDto = Statistics.calculate(
                    5,
                    [algorithm],
                    [f1_tool_evaluator, f1_tool_evaluator_strict],
                    [],
                    count_of_sessions,
                    gold_session,
                    prompt,
                    reference,
                    baseline_logger,
                    None,
                    subdirectory,
                    True
                )
            else:
                self.logger.info("Start evaluating memory statistics")
                statistics: StatisticsDto = Statistics.calculate(
                    5,
                    [algorithm],
                    [f1_tool_evaluator, f1_tool_evaluator_strict],
                    past_interactions,
                    count_of_sessions,
                    gold_session,
                    prompt,
                    reference,
                    memory_logger,
                    None,
                    subdirectory,
                    True
                )

        Statistics.print_statistics(
            statistics
        )

    @staticmethod
    def get_statistics_by_directory_with_logs(path: Path | str) -> StatisticsDto:
        records: list[BaseRecord] = []
        for f in [
            "BaseMemoryBank",
            "BaseRecsum",
            "FullBaseline",
            "LastBaseline",
            "RagMemoryBank",
            "RagRecsum"
        ]:
            folder = Path(path) / f
            for path in folder.glob("*.json"):
                print(path)
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                    if obj.get("memory") is None:
                        record = BaseRecord.from_dict(obj)
                    else:
                        record = MemoryRecord.from_dict(obj)
                    records.append(record)
        return Statistics.calculate_by_logs(
            count_of_launches=10,
            metrics=records
        )

    @staticmethod
    def get_spent_tokens_count(logs: BaseRecord, model: OpenAIModels) -> TokenInfo:
        prompt = str(logs.query)
        response = str(logs.response)
        input_tokens = Runner.__count_tokens(prompt)
        output_tokens = Runner.__count_tokens(response)
        input_price = input_tokens * MODEL_PRICES[model].input_per_million / 1_000_000
        output_price = output_tokens * MODEL_PRICES[model].output_per_million / 1_000_000
        return TokenInfo(
            model=model,
            price=MODEL_PRICES[model],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_price=input_price,
            output_price=output_price,
            total_price=input_price + output_price
        )

    def __execute_query_and_reference(self, past_interactions: Session) -> QueryAndReference:
        reference: list[BaseBlock] = []
        query: BaseBlock | None = None
        for i in range(len(past_interactions.messages) - 1, -1, -1):
            if past_interactions.messages[i].role == "USER" and past_interactions.messages[i].content != "":
                self.logger.info(f"User founded {i}")
                self.logger.info(f"User message: {past_interactions.messages[i].content}")
                query = past_interactions.messages[i]
                break
            else:
                reference.append(past_interactions.messages[i])

        assert query is not None, "User's query is not founded."

        reference = reference[::-1]

        return QueryAndReference(
            query=query,
            reference=reference
        )

    @staticmethod
    def tokens():
        Statistics.print_statistics(
            Runner.get_statistics_by_directory_with_logs("logs/memory/2025-11-15T21:57:10.007355")
        )

        path = Path(
            "/Users/mikhailkharlamov/Documents/.../RecapKt/src/benchmarking/tool_metrics/logs/memory/2025-11-18T17:40:37.936141")
        for directory in [
            "BaseMemoryBank",
            "BaseRecsum",
            "FullBaseline",
            "LastBaseline",
            "RagMemoryBank",
            "RagRecsum"
        ]:
            folder = path / directory
            ps = []
            for p in folder.glob("*.json"):
                ps.append(p)
            ps.sort()

            for i in range(len(ps)):
                print(i)
                p = ps[i]
                with p.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("memory") is None:
                    record = BaseRecord.from_dict(d)
                else:
                    record = MemoryRecord.from_dict(d)
                tokens_info = Runner.get_spent_tokens_count(record, OpenAIModels.GPT_4_O_MINI)
                p_n = p.resolve().parent / f"{p.stem}_tokens.json"
                with p_n.open("w", encoding="utf-8") as f:
                    json.dump(tokens_info.to_dict(encode_json=True), f, indent=4)

    @staticmethod
    def build_graph(
            graph_types: list[type[GraphBuilder]],
            directories: list[Path | str],
    ):
        algs: list[AlgorithmStatistics] = []
        for fold in [1, 3, 5, 7, 9, 11, 13, 15]:
            ps: list[Path] = []
            path = Path(
                "/Users/mikhailkharlamov/Documents/.../RecapKt/src/benchmarking/tool_metrics/logs/memory")
            for directory in directories:
                folder = path / directory / f"{fold}"
                for p in folder.glob("*.json"):
                    ps.append(p)
                ps.sort()

            r: list[BaseRecord] = []
            for p in ps:
                with p.open("r", encoding="utf8") as f:
                    j = json.load(f)
                    if "memory" in j:
                        data = MemoryRecord.from_dict(j)
                    else:
                        data = BaseRecord.from_dict(j)
                    r.append(data)

            stats = Statistics.calculate_by_logs(fold, r)
            algs.extend(stats.algorithms)
            Statistics.print_statistics(stats)
            #for graph in graph_types:
            #    graph.build(stats, "", f". {fold}")
        f1_algs = []
        f1_strict = []
        for graph in graph_types:
            for alg in algs:
                if alg.metric == MetricType.F1_TOOL:
                    f1_algs.append(alg)
                else:
                    f1_strict.append(alg)

            graph.build(StatisticsDto(algorithms=f1_algs), "", " lax")
            graph.build(StatisticsDto(algorithms=f1_strict), "", " strict")

    def __prepare_query_for_the_first_stage(self, query: BaseBlock) -> str:
        template = self.env.get_template("first_stage.j2")
        rendered_prompt = template.render(query=query.content)
        return rendered_prompt

    @staticmethod
    def __print_metrics(log_records: list[dict[str, Any]]) -> None:
        for record in log_records:
            print(f"System: {record['system']}")
            print("Metrics:")
            for metric in record.get("metric"):
                print(f"  - {metric.get("metric_name")}: {metric.get("metric_value")}")
            print("\n")

    @staticmethod
    def __count_tokens(text: str) -> int:
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(text)
        return len(tokens)


if __name__ == "__main__":
    configure_logs(loglevel=logging.INFO)

    """runner = Runner()


    print(sys.argv)
    runner.run(sys.argv[1])"""

    Runner.build_graph(
        [GeneralTrends],
        [
            "BaseMemoryBank",
            "BaseRecsum",
            "FullBaseline",
            "LastBaseline",
            "RagMemoryBank",
            "RagRecsum",
        ],
    )
    """for directory in [
            "BaseMemoryBank",
            "BaseRecsum",
            "FullBaseline",
            "LastBaseline",
            "RagMemoryBank",
            "RagRecsum",
        ]:
        Runner.build_graph(
            [TrendsWithQuantiles],
            [directory],
        )"""


