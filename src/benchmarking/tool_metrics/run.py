import json
import logging
import os
import sys

from pathlib import Path

import tiktoken

from jinja2 import Environment, FileSystemLoader
from load_dotenv import load_dotenv

from src.benchmarking.agent.dialog_short_tools import DialogueWithShortTools
from src.benchmarking.agent.dialog_with_weights import DialogueWithWeights
from src.benchmarking.agent.dialogue_baseline import DialogueBaseline
from src.benchmarking.baseline_logger import BaselineLogger
from src.benchmarking.memory_logger import MemoryLogger
from src.benchmarking.models.dtos import (
    MODEL_PRICES,
    AlgorithmStatistics,
    BaseRecord,
    DividedSession,
    MemoryRecord,
    StatisticsDto,
    TokenInfo,
)
from src.benchmarking.models.enums import AlgorithmDirectory, AlgorithmName, MetricType
from src.benchmarking.tool_metrics.evaluators.f1_tool_evaluator import F1ToolEvaluator
from src.benchmarking.tool_metrics.graphs.general_trends import GeneralTrends
from src.benchmarking.tool_metrics.graphs.graph_builder import GraphBuilder
from src.benchmarking.tool_metrics.load_session import Loader
from src.benchmarking.tool_metrics.statistics_calc import Statistics
from src.summarize_algorithms.core.dialogue import Dialogue
from src.summarize_algorithms.core.models import (
    BaseBlock,
    OpenAIModels,
    Session,
)
from src.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueSystem,
)
from src.summarize_algorithms.recsum.dialogue_system import RecsumDialogueSystem
from src.utils.configure_logs import configure_logs

load_dotenv()

BASE_DATA_PATH = os.getenv("BASE_DATA_PATH", "")
LOGS_PATH = os.getenv("LOGS_PATH", "")
JSON_FILE_TEMPLATE: str = "*.json"


class Runner:
    """
    Orchestrates tool-metrics benchmarking runs.

    Responsibilities:
    - loads past sessions + a "gold" session from `BASE_DATA_PATH`
    - runs a selected dialogue system/baseline
    - evaluates outputs via `BaseEvaluator` implementations (e.g. `F1ToolEvaluator`)
    - optionally builds graphs from saved logs
    """

    def __init__(self, templates_dir: str = "prompts") -> None:
        self._logger = logging.getLogger()
        self._env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=True,
            trim_blocks=True
        )

        self._baseline_logger = BaselineLogger()
        self._memory_logger = MemoryLogger()

    def run(self, name: str) -> None:
        """
        Run a full benchmark sweep for a given algorithm name.

        Loads input sessions from `BASE_DATA_PATH`, selects the algorithm/baseline, evaluates it with tool metrics,
        and prints aggregated statistics.

        :param name: algorithm name (see `AlgorithmName`).
        :return: None
        """
        algorithm: Dialogue = Runner.__init_algorithm(AlgorithmName(name))

        self._logger.info("Start parsing session")
        past_interactions: list[Session] = []

        path_data_type_1: Path = Path(BASE_DATA_PATH) / "data_type_1"
        for file in path_data_type_1.glob(JSON_FILE_TEMPLATE):
            past_interactions.append(
                Loader.load_session_data_type_1(file)
            )

        path_data_type_2: Path = Path(BASE_DATA_PATH)  / "data_type_2"
        for file in path_data_type_2.glob(JSON_FILE_TEMPLATE):
            past_interactions.append(
                Loader.load_session_data_type_2(file)
            )

        gold_session: Session = Loader.load_session_data_type_1(Path(BASE_DATA_PATH) / "gold_session.json")

        divided_session: DividedSession = self.__divide_session(gold_session)
        reference, session = divided_session.reference, divided_session.past_interactions
        prompt = self.__prepare_system_prompt()

        f1_tool_evaluator_strict = F1ToolEvaluator("strict")
        f1_tool_evaluator = F1ToolEvaluator()

        for count_of_sessions in [1, 3, 5, 7, 9, 11, 13, 15]:
            subdirectory: Path = Path(str(count_of_sessions))

            if name in ("full_baseline", "short_tools", "weights"):
                self._logger.info("Start evaluating full baseline statistics")
                statistics: StatisticsDto = Statistics.calculate(
                    5,
                    [algorithm],
                    [f1_tool_evaluator, f1_tool_evaluator_strict],
                    past_interactions,
                    count_of_sessions,
                    Session(session),
                    prompt,
                    reference,
                    self._baseline_logger,
                    subdirectory,
                    None,
                    True
                )
            elif name == "last_baseline":
                self._logger.info("Start evaluating last baseline statistics")
                statistics = Statistics.calculate(
                    5,
                    [algorithm],
                    [f1_tool_evaluator, f1_tool_evaluator_strict],
                    [],
                    count_of_sessions,
                    Session(session),
                    prompt,
                    reference,
                    self._baseline_logger,
                    subdirectory,
                    None,
                    True
                )
            else:
                self._logger.info("Start evaluating memory statistics")
                statistics = Statistics.calculate(
                    5,
                    [algorithm],
                    [f1_tool_evaluator, f1_tool_evaluator_strict],
                    past_interactions,
                    count_of_sessions,
                    Session(session),
                    prompt,
                    reference,
                    self._memory_logger,
                    subdirectory,
                    None,
                    True
                )

        Statistics.print_statistics(
            statistics
        )

    @staticmethod
    def __init_algorithm(name: AlgorithmName) -> Dialogue:
        if name.value == "base_recsum":
            return RecsumDialogueSystem(embed_code=False, embed_tool=False, system_name="BaseRecsum")
        elif name.value == "base_memory_bank":
            return MemoryBankDialogueSystem(embed_code=False, embed_tool=False, system_name="BaseMemoryBank")
        elif name.value == "rag_recsum":
            return RecsumDialogueSystem(embed_code=True, embed_tool=True, system_name="RagRecsum")
        elif name.value == "rag_memory_bank":
            return MemoryBankDialogueSystem(embed_code=True, embed_tool=True, system_name="RagMemoryBank")
        elif name.value == "full_baseline":
            return DialogueBaseline("FullBaseline")
        elif name.value == "short_tools":
            return DialogueWithShortTools("ShortTools")
        elif name.value == "weights":
            return DialogueWithWeights("Weights")
        else:
            return DialogueBaseline("LastBaseline")

    @staticmethod
    def get_statistics_by_directory_with_logs(path: Path | str) -> StatisticsDto:
        records: list[BaseRecord] = []
        for alg_folder in [
            "BaseMemoryBank",
            "BaseRecsum",
            "FullBaseline",
            "LastBaseline",
            "RagMemoryBank",
            "RagRecsum"
        ]:
            folder = Path(path) / alg_folder
            for path in folder.glob(JSON_FILE_TEMPLATE):
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

    def __divide_session(self, session: Session) -> DividedSession:
        past_interactions: list[BaseBlock] = []
        reference: list[BaseBlock] = []
        query: BaseBlock | None = None
        is_query_found: bool = False
        for i in range(len(session.messages) - 1, -1, -1):
            if is_query_found:
                past_interactions.append(session.messages[i])
            elif session.messages[i].role == "USER" and session.messages[i].content != "":
                self._logger.info(f"User founded {i}")
                self._logger.info(f"User message: {session.messages[i].content}")
                query = session.messages[i]
                is_query_found = True
            else:
                reference.append(session.messages[i])

        assert query is not None, "User's query is not founded."

        reference = reference[::-1]
        past_interactions = past_interactions[::-1]
        past_interactions.append(query)

        return DividedSession(
            reference=reference,
            past_interactions=past_interactions,
        )

    @staticmethod
    def tokens() -> None:
        Statistics.print_statistics(
            Runner.get_statistics_by_directory_with_logs(LOGS_PATH)
        )

        path = Path(LOGS_PATH)
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
            for p in folder.glob(JSON_FILE_TEMPLATE):
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
    ) -> None:
        algs: list[AlgorithmStatistics] = []
        for fold in [1, 3, 5, 7, 9, 11, 13, 15]:
            ps: list[Path] = []
            path = Path(LOGS_PATH)
            for directory in directories:
                folder = path / directory / f"{fold}"
                for p in folder.glob(JSON_FILE_TEMPLATE):
                    ps.append(p)
                ps.sort()

            r: list[BaseRecord] = []
            for p in ps:
                with p.open("r", encoding="utf8") as f:
                    j = json.load(f)
                    if "memory" in j:
                        data: BaseRecord = MemoryRecord.from_dict(j)
                    else:
                        data = BaseRecord.from_dict(j)
                    r.append(data)

            stats = Statistics.calculate_by_logs(fold, r)
            algs.extend(stats.algorithms)
            Statistics.print_statistics(stats)

        f1_algs = []
        f1_strict = []
        for graph in graph_types:
            for alg in algs:
                if alg.metric == MetricType.F1_TOOL:
                    f1_algs.append(alg)
                else:
                    f1_strict.append(alg)

            graph.build(StatisticsDto(algorithms=f1_algs), "", "nonstrict")
            graph.build(StatisticsDto(algorithms=f1_strict), "", "strict")

    def __prepare_system_prompt(self) -> str:
        template = self._env.get_template("first_stage.j2")
        rendered_prompt = template.render()
        return rendered_prompt

    @staticmethod
    def __count_tokens(text: str) -> int:
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(text)
        return len(tokens)


if __name__ == "__main__":
    configure_logs(loglevel=logging.INFO)

    runner = Runner()
    runner.run(sys.argv[1])

    Runner.build_graph(
        [GeneralTrends],
        [
            AlgorithmDirectory.FULL_BASELINE.value,
            AlgorithmDirectory.LAST_BASELINE.value,
            AlgorithmDirectory.RAG_MEMORY_BANK.value,
            AlgorithmDirectory.RAG_RECSUM.value,
            AlgorithmDirectory.RAG_MEMORY_BANK.value,
            AlgorithmDirectory.BASE_MEMORY_BANK.value,
            AlgorithmDirectory.BASE_RECSUM.value,
            AlgorithmDirectory.WEIGHTS.value,
            AlgorithmDirectory.SHORT_TOOLS.value,
        ]
    )
