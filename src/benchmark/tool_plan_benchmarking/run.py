import argparse
import json
import logging
import os

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import tiktoken

from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import messages_from_dict
from langchain_openai import ChatOpenAI
from load_dotenv import load_dotenv
from pydantic import SecretStr

from src.algorithms.dialogue import Dialogue
from src.algorithms.simple_algorithms.dialog_short_tools import DialogueWithShortTools
from src.algorithms.simple_algorithms.dialog_with_weights import DialogueWithWeights
from src.algorithms.simple_algorithms.dialogue_baseline import DialogueBaseline
from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    OpenAIModels,
    Session,
)
from src.algorithms.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueSystem,
)
from src.algorithms.summarize_algorithms.recsum.dialogue_system import (
    RecsumDialogueSystem,
)
from src.benchmark.logger.base_logger import BaseLogger
from src.benchmark.logger.baseline_logger import BaselineLogger
from src.benchmark.logger.memory_logger import MemoryLogger
from src.benchmark.models.dtos import (
    MODEL_PRICES,
    BaseRecord,
    DividedSession,
    MemoryRecord,
    TokenInfo,
)
from src.benchmark.models.enums import AlgorithmDirectory, AlgorithmName, MetricType
from src.benchmark.tool_plan_benchmarking.evaluators.f1_tool_evaluator import (
    F1ToolEvaluator,
)
from src.benchmark.tool_plan_benchmarking.graphs.general_trends import GeneralTrends
from src.benchmark.tool_plan_benchmarking.graphs.graph_builder import GraphBuilder
from src.benchmark.tool_plan_benchmarking.load_session import Loader
from src.benchmark.tool_plan_benchmarking.statistics.dtos import (
    AlgorithmStatistics,
    StatisticsDto,
)
from src.benchmark.tool_plan_benchmarking.statistics.statistics import Statistics
from src.utils.configure_logs import configure_logs

load_dotenv()

BASE_DATA_PATH = os.getenv("BASE_DATA_PATH", "")
LOGS_PATH = os.getenv("LOGS_PATH", Path(__file__).resolve().parent / "logs" / "memory")

JSON_FILE_TEMPLATE: str = "*.json"
TOKENS_FILE_SUFFIX: str = "_tokens.json"
TOKENS_AVERAGED_FILENAME: str = "tokens_averaged.json"

ALGORITHM_DIRS_FOR_TOKENS: list[str] = [
    "BaseMemoryBank",
    "BaseRecsum",
    "FullBaseline",
    "LastBaseline",
    "RagMemoryBank",
    "RagRecsum",
    "ShortTools",
    "Weights",
]

SESSION_COUNTS_FOR_TOKENS: list[str] = ["1", "3", "5", "7", "9", "11", "13", "15"]
RUNS_PER_FOLDER_FOR_AVERAGING: int = 5


class Runner:
    """
    Orchestrates tool-metrics benchmark runs.

    Responsibilities:
    - loads past sessions + a "gold" session from `BASE_DATA_PATH`
    - runs a selected dialogue system/baseline
    - evaluates outputs via `BaseEvaluator` implementations (e.g. `F1ToolEvaluator`)
    - optionally builds graphs from saved logs
    """

    def __init__(self, templates_dir: str = "prompts") -> None:
        self._logger = logging.getLogger()
        self._env = Environment(
            loader=FileSystemLoader(templates_dir), autoescape=True, trim_blocks=True
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
            past_interactions.append(Loader.load_session_data_type_1(file))

        path_data_type_2: Path = Path(BASE_DATA_PATH) / "data_type_2"
        for file in path_data_type_2.glob(JSON_FILE_TEMPLATE):
            past_interactions.append(Loader.load_session_data_type_2(file))

        gold_session: Session = Loader.load_session_data_type_1(
            Path(BASE_DATA_PATH) / "gold_session.json"
        )

        divided_session: DividedSession = self.__divide_session(gold_session)
        reference, session = (
            divided_session.reference,
            divided_session.past_interactions,
        )

        # The algorithms expect `system_prompt` argument to contain the latest user request.
        prompt = session[-1].content if len(session) > 0 else ""

        f1_tool_evaluator_strict = F1ToolEvaluator("strict")
        f1_tool_evaluator_arguments_similarity = F1ToolEvaluator("arguments_similarity")
        f1_tool_evaluator = F1ToolEvaluator()

        for count_of_sessions in [15]:
            subdirectory: Path = Path(str(count_of_sessions))

            if name in ("full_baseline", "short_tools", "weights"):
                self._logger.info("Start evaluating full baseline statistics")
                statistics: StatisticsDto = Statistics.calculate(
                    5,
                    [algorithm],
                    [
                        f1_tool_evaluator,
                        f1_tool_evaluator_strict,
                        f1_tool_evaluator_arguments_similarity,
                    ],
                    past_interactions,
                    count_of_sessions,
                    Session(session),
                    prompt,
                    reference,
                    self._baseline_logger,
                    subdirectory,
                    None,
                    True,
                )
            elif name == "last_baseline":
                self._logger.info("Start evaluating last baseline statistics")
                statistics = Statistics.calculate(
                    5,
                    [algorithm],
                    [
                        f1_tool_evaluator,
                        f1_tool_evaluator_strict,
                        f1_tool_evaluator_arguments_similarity,
                    ],
                    [],
                    count_of_sessions,
                    Session(session),
                    prompt,
                    reference,
                    self._baseline_logger,
                    subdirectory,
                    None,
                    True,
                )
            else:
                self._logger.info("Start evaluating memory statistics")
                statistics = Statistics.calculate(
                    5,
                    [algorithm],
                    [
                        f1_tool_evaluator,
                        f1_tool_evaluator_strict,
                        f1_tool_evaluator_arguments_similarity,
                    ],
                    past_interactions,
                    count_of_sessions,
                    Session(session),
                    prompt,
                    reference,
                    self._memory_logger,
                    subdirectory,
                    None,
                    True,
                )

        Statistics.print_statistics(statistics)

    def evaluate_by_logs(
        self,
        name: str,
        logs_path: Path | str = LOGS_PATH,
        iteration: int | None = None,
    ) -> None:
        """Append newly-added metrics to existing log JSONs and print updated statistics.

        This mode does *not* re-run the dialogue system. It:
        - loads `gold_session.json` from `BASE_DATA_PATH` to build the reference trace
        - re-evaluates the latest saved logs under `logs_path` (in-place)

        CLI usage (see `__main__` below):
            python -m src.benchmark.tool_plan_benchmarking.run <algo> --eval-by-logs [--logs-path PATH] [--iteration N]
        """
        algorithm: Dialogue = Runner.__init_algorithm(AlgorithmName(name))

        gold_session: Session = Loader.load_session_data_type_1(
            Path(BASE_DATA_PATH) / "gold_session.json"
        )
        divided_session: DividedSession = self.__divide_session(gold_session)
        reference = divided_session.reference

        f1_tool_evaluator_strict = F1ToolEvaluator("strict")
        f1_tool_evaluator_arguments_similarity = F1ToolEvaluator("arguments_similarity")
        f1_tool_evaluator = F1ToolEvaluator("nonstrict")

        if name in ("full_baseline", "short_tools", "weights"):
            logger: BaseLogger = BaselineLogger()
        else:
            logger = MemoryLogger()

        for count_of_sessions in [1, 3, 5, 7, 9, 11, 13, 15]:
            print(count_of_sessions)
            subdirectory: Path = Path(str(count_of_sessions))
            self._logger.info("Start evaluating by logs (fold=%s)", count_of_sessions)

            statistics = Statistics.calculate_with_new_metrics_by_logs(
                algorithms=[algorithm],
                evaluator_functions=[
                    f1_tool_evaluator,
                    f1_tool_evaluator_strict,
                    f1_tool_evaluator_arguments_similarity,
                ],
                reference=reference,
                logger=logger,
                logs_path=logs_path,
                subdirectory=subdirectory,
                iteration=iteration,
                normalize=False,
            )

            Statistics.print_statistics(statistics)

    @staticmethod
    def __init_algorithm(name: AlgorithmName) -> Dialogue:
        if name.value == "base_recsum":
            return RecsumDialogueSystem(
                embed_code=False, embed_tool=False, system_name="BaseRecsum"
            )
        elif name.value == "base_memory_bank":
            return MemoryBankDialogueSystem(
                embed_code=False, embed_tool=False, system_name="BaseMemoryBank"
            )
        elif name.value == "rag_recsum":
            return RecsumDialogueSystem(
                embed_code=True, embed_tool=True, system_name="RagRecsum"
            )
        elif name.value == "rag_memory_bank":
            return MemoryBankDialogueSystem(
                embed_code=True, embed_tool=True, system_name="RagMemoryBank"
            )
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
            "RagRecsum",
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
        return Statistics.calculate_by_logs(count_of_launches=10, metrics=records)

    @staticmethod
    def get_spent_tokens_count(logs: BaseRecord, model: OpenAIModels) -> TokenInfo:
        # IMPORTANT: prompt tokens are counted from `prepared_messages` (not from `logs.query`).
        input_tokens = Runner.__count_prepared_messages_tokens(
            logs.prepared_messages, model
        )

        response = str(logs.response)
        output_tokens = Runner.__count_tokens(response)

        input_price = input_tokens * MODEL_PRICES[model].input_per_million / 1_000_000
        output_price = (
            output_tokens * MODEL_PRICES[model].output_per_million / 1_000_000
        )
        return TokenInfo(
            model=model,
            price=MODEL_PRICES[model],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_price=input_price,
            output_price=output_price,
            total_price=input_price + output_price,
        )

    @staticmethod
    def __count_prepared_messages_tokens(
        raw_prepared_messages: list[dict[str, Any]],
        model: OpenAIModels,
    ) -> int:
        if not raw_prepared_messages:
            return 0

        try:
            # Logs store `BaseMessage.model_dump(mode="json")` output.
            # LangChain expects: {"type": <...>, "data": { ...message fields... }}
            wrapped: list[dict[str, Any]] = []
            for raw in raw_prepared_messages:
                if not isinstance(raw, dict):
                    continue

                msg_type = raw.get("type")
                if not isinstance(msg_type, str):
                    continue

                wrapped.append(
                    {
                        "type": msg_type,
                        "data": {k: v for k, v in raw.items() if k != "type"},
                    }
                )

            prepared_messages: list[BaseMessage] = messages_from_dict(wrapped)

            # `get_num_tokens_from_messages` is offline (no API call) and uses model-specific tokenization.
            llm = ChatOpenAI(model=model.value, api_key=SecretStr("DUMMY"))
            return llm.get_num_tokens_from_messages(prepared_messages)
        except Exception:
            logging.exception(
                "Failed to count tokens from prepared_messages via LangChain; falling back to tiktoken on JSON dump."
            )
            return Runner.__count_tokens(
                json.dumps(raw_prepared_messages, ensure_ascii=False)
            )

    def __divide_session(self, session: Session) -> DividedSession:
        past_interactions: list[BaseBlock] = []
        reference: list[BaseBlock] = []
        query: BaseBlock | None = None
        is_query_found: bool = False
        for i in range(len(session.messages) - 1, -1, -1):
            if is_query_found:
                past_interactions.append(session.messages[i])
            elif (
                session.messages[i].role == "USER" and session.messages[i].content != ""
            ):
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
        for directory in ALGORITHM_DIRS_FOR_TOKENS:
            for count_of_sessions in SESSION_COUNTS_FOR_TOKENS:
                folder = path / directory / count_of_sessions
                if not folder.exists():
                    continue

                ps: list[Path] = [
                    p
                    for p in folder.glob(JSON_FILE_TEMPLATE)
                    if not p.name.endswith(TOKENS_FILE_SUFFIX)
                    and p.name != TOKENS_AVERAGED_FILENAME
                ]
                ps.sort()

                for i, p in enumerate(ps):
                    print(i)
                    with p.open("r", encoding="utf-8") as f:
                        d = json.load(f)

                    if d.get("memory") is None:
                        record = BaseRecord.from_dict(d)
                    else:
                        record = MemoryRecord.from_dict(d)

                    tokens_info = Runner.get_spent_tokens_count(
                        record, OpenAIModels.GPT_4_O_MINI
                    )
                    p_n = p.resolve().parent / f"{p.stem}{TOKENS_FILE_SUFFIX}"
                    with p_n.open("w", encoding="utf-8") as f:
                        json.dump(tokens_info.to_dict(encode_json=True), f, indent=4)

    @staticmethod
    def tokens_averaged(model: OpenAIModels = OpenAIModels.GPT_4_O_MINI) -> None:
        path = Path(LOGS_PATH)
        result: dict[str, dict[str, dict[str, Any]]] = {}

        for directory in ALGORITHM_DIRS_FOR_TOKENS:
            directory_result: dict[str, dict[str, Any]] = {}

            for count_of_sessions in SESSION_COUNTS_FOR_TOKENS:
                folder = path / directory / count_of_sessions
                if not folder.exists():
                    continue

                ps: list[Path] = [
                    p
                    for p in folder.glob(JSON_FILE_TEMPLATE)
                    if not p.name.endswith(TOKENS_FILE_SUFFIX)
                    and p.name != TOKENS_AVERAGED_FILENAME
                ]
                ps.sort()

                ps = ps[-RUNS_PER_FOLDER_FOR_AVERAGING:]

                token_infos: list[TokenInfo] = []
                for p in ps:
                    with p.open("r", encoding="utf-8") as f:
                        d = json.load(f)

                    if d.get("memory") is None:
                        record = BaseRecord.from_dict(d)
                    else:
                        record = MemoryRecord.from_dict(d)

                    token_infos.append(Runner.get_spent_tokens_count(record, model))

                if not token_infos:
                    continue

                n = Decimal(len(token_infos))

                sum_input_tokens = sum(t.input_tokens for t in token_infos)
                sum_output_tokens = sum(t.output_tokens for t in token_infos)

                sum_input_price = sum(
                    (t.input_price for t in token_infos), start=Decimal("0")
                )
                sum_output_price = sum(
                    (t.output_price for t in token_infos), start=Decimal("0")
                )
                sum_total_price = sum(
                    (t.total_price for t in token_infos), start=Decimal("0")
                )

                avg_input_tokens = int(
                    (Decimal(sum_input_tokens) / n).to_integral_value(
                        rounding=ROUND_HALF_UP
                    )
                )
                avg_output_tokens = int(
                    (Decimal(sum_output_tokens) / n).to_integral_value(
                        rounding=ROUND_HALF_UP
                    )
                )

                avg_input_price = sum_input_price / n
                avg_output_price = sum_output_price / n
                avg_total_price = sum_total_price / n

                averaged = TokenInfo(
                    model=model,
                    price=MODEL_PRICES[model],
                    input_tokens=avg_input_tokens,
                    output_tokens=avg_output_tokens,
                    input_price=avg_input_price,
                    output_price=avg_output_price,
                    total_price=avg_total_price,
                )

                directory_result[count_of_sessions] = averaged.to_dict(encode_json=True)

            if directory_result:
                result[directory] = directory_result

        output_path = path / TOKENS_AVERAGED_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    @staticmethod
    def build_graph(
        graph_types: list[type[GraphBuilder]],
        directories: list[Path | str],
        normalize: bool = False,
    ) -> None:
        path = Path(LOGS_PATH)
        if not path.exists():
            logging.getLogger(__name__).warning(
                "LOGS_PATH does not exist: %s. Set env LOGS_PATH or place logs under the default path.",
                path,
            )
            return

        algs: list[AlgorithmStatistics] = []
        for fold in [1, 3, 5, 7, 9, 11, 13, 15]:
            ps: list[Path] = []
            for directory in sorted({str(d) for d in directories}):
                folder = path / directory / f"{fold}"
                if not folder.exists():
                    continue
                ps.extend(folder.glob(JSON_FILE_TEMPLATE))

            ps.sort()

            if not ps:
                logging.getLogger(__name__).info(
                    "No log files found for fold=%s under %s. Skipping fold.",
                    fold,
                    path,
                )
                continue

            r: list[BaseRecord] = []
            for p in ps:
                if p.name.endswith(TOKENS_FILE_SUFFIX):
                    continue
                with p.open("r", encoding="utf8") as f:
                    j = json.load(f)
                    if "memory" in j:
                        data: BaseRecord = MemoryRecord.from_dict(j)
                    else:
                        data = BaseRecord.from_dict(j)
                    r.append(data)

            stats = Statistics.calculate_by_logs(fold, r, normalize=normalize)
            algs.extend(stats.algorithms)
            Statistics.print_statistics(stats)

        for graph in graph_types:
            f1_nonstrict = [alg for alg in algs if alg.metric == MetricType.F1_TOOL]
            f1_arguments_similarity = [
                alg
                for alg in algs
                if alg.metric == MetricType.F1_TOOL_ARGUMENTS_SIMILARITY
            ]
            f1_strict = [alg for alg in algs if alg.metric == MetricType.F1_TOOL_STRICT]

            graphs_dir = path / "graphs"
            graphs_dir.mkdir(parents=True, exist_ok=True)

            graph.build(
                StatisticsDto(algorithms=f1_nonstrict),
                str(graphs_dir / f"{graph.__name__}_nonstrict.png"),
                "nonstrict",
            )
            graph.build(
                StatisticsDto(algorithms=f1_strict),
                str(graphs_dir / f"{graph.__name__}_strict.png"),
                "strict",
            )
            graph.build(
                StatisticsDto(algorithms=f1_arguments_similarity),
                str(graphs_dir / f"{graph.__name__}_arguments_similarity.png"),
                "arguments_similarity",
            )

    def __prepare_system_prompt(self) -> str:
        template = self._env.get_template("first_stage.j2")
        rendered_prompt = template.render()
        return rendered_prompt

    @staticmethod
    def __count_tokens(text: str) -> int:
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(text)
        return len(tokens)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tool-plan benchmarking runner")
    parser.add_argument(
        "name",
        help=f"Algorithm name. Allowed: {[a.value for a in AlgorithmName]}",
    )
    parser.add_argument(
        "--eval-by-logs",
        action="store_true",
        help="Do not run algorithms; instead, append newly-added metrics by re-evaluating saved logs in-place.",
    )
    parser.add_argument(
        "--logs-path",
        default=str(LOGS_PATH),
        help="Root directory with logs (default: env LOGS_PATH or tool default).",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="If set, evaluates only the newest log file with this iteration value.",
    )
    return parser


if __name__ == "__main__":
    configure_logs(loglevel=logging.INFO)

    runner = Runner()
    for name in [
        AlgorithmName.FULL_BASELINE.value,
        AlgorithmName.LAST_BASELINE.value,
        AlgorithmName.RAG_MEMORY_BANK.value,
        AlgorithmName.RAG_RECSUM.value,
        AlgorithmName.BASE_MEMORY_BANK.value,
        AlgorithmName.BASE_RECSUM.value,
        AlgorithmName.WEIGHTS.value,
        AlgorithmName.SHORT_TOOLS.value,
    ]:
        print(name)
        runner.evaluate_by_logs(name)

    Runner.build_graph(
        [GeneralTrends],
        [
            AlgorithmDirectory.FULL_BASELINE.value,
            AlgorithmDirectory.LAST_BASELINE.value,
            AlgorithmDirectory.RAG_MEMORY_BANK.value,
            AlgorithmDirectory.RAG_RECSUM.value,
            AlgorithmDirectory.BASE_MEMORY_BANK.value,
            AlgorithmDirectory.BASE_RECSUM.value,
            AlgorithmDirectory.WEIGHTS.value,
            AlgorithmDirectory.SHORT_TOOLS.value,
        ],
        normalize=False,
    )

    """args = _build_arg_parser().parse_args()

    runner = Runner()
    if args.eval_by_logs:
        runner.evaluate_by_logs(args.name, logs_path=args.logs_path, iteration=args.iteration)
    else:
        runner.run(args.name)

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
        ],
        normalize=False,
    )

    #Runner.tokens()

    Runner.tokens_averaged()"""
