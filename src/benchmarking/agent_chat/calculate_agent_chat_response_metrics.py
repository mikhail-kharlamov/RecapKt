import itertools
import random

from dataclasses import dataclass, field

from src.benchmarking.agent_chat.deserialize_agent_chat import ChatDataset
from src.benchmarking.baseline import DialogueBaseline
from src.benchmarking.llm_evaluation import (
    ComparisonResult,
    LLMChatAgentEvaluation,
    SingleChatAgentResult,
)
from src.summarize_algorithms.core.models import Session
from src.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueSystem,
)
from src.summarize_algorithms.recsum.dialogue_system import RecsumDialogueSystem


@dataclass
class SingleResult:
    correctness: list[int] = field(default_factory=list)
    clarity: list[int] = field(default_factory=list)
    context_handling: list[int] = field(default_factory=list)


@dataclass
class PairwiseResult:
    base_recsum: int = 0
    rag_recsum: int = 0
    base_memory_bank: int = 0
    rag_memory_bank: int = 0
    full_baseline: int = 0
    last_baseline: int = 0


class CalculateAgentChatResponseMetrics:
    def __init__(self) -> None:
        self.dataset = ChatDataset.from_file()
        self.llm_scorer = LLMChatAgentEvaluation()
        self.message_count = 0

        self.base_recsum_single_result = SingleResult()
        self.rag_recsum_single_result = SingleResult()
        self.base_memory_bank_single_result = SingleResult()
        self.rag_memory_bank_single_result = SingleResult()
        self.full_sessions_baseline_single_result = SingleResult()
        self.last_session_baseline_single_result = SingleResult()

        self.pairwise_result = PairwiseResult()

        self.base_recsum = RecsumDialogueSystem(embed_code=False, embed_tool=False)
        self.rag_recsum = RecsumDialogueSystem(embed_code=True, embed_tool=True)

        self.base_memory_bank = MemoryBankDialogueSystem(
            embed_code=False, embed_tool=False
        )
        self.rag_memory_bank = MemoryBankDialogueSystem(
            embed_code=True, embed_tool=True
        )

        self.full_baseline = DialogueBaseline()
        self.last_baseline = DialogueBaseline()

    def calculate(self) -> None:
        dialogue = self.dataset.sessions
        for i in range(len(dialogue)):
            print(f"Processing dialogue {i + 1}/{len(dialogue)}")
            self._process(dialogue[: i + 1])

    def _process(self, sessions: list[Session]) -> None:
        last_session = sessions[-1]
        query = ""
        for i in range(len(last_session.messages) - 1, -1, -1):
            if last_session.messages[i].role == "user":
                query = last_session.messages[i].content
                break

        dialogue_context = ""
        for i in range(len(sessions)):
            dialogue_context += f"Session: {i}" + str(sessions[i]) + "\n\n"

        base_recsum_response = self.base_recsum.process_dialogue(
            sessions, query
        ).response
        rag_recsum_response = self.rag_recsum.process_dialogue(sessions, query).response
        base_memory_bank_response = self.base_memory_bank.process_dialogue(
            sessions, query
        ).response
        rag_memory_bank_response = self.rag_memory_bank.process_dialogue(
            sessions, query
        ).response
        full_sessions_baseline_response = self.full_baseline.process_dialogue(
            sessions, query
        )
        last_session_baseline_response = self.last_baseline.process_dialogue(
            [sessions[-1]], query
        )

        base_recsum_single_score = self.llm_scorer.evaluate_single(
            dialogue_context=dialogue_context, assistant_answer=base_recsum_response
        )
        rag_recsum_single_score = self.llm_scorer.evaluate_single(
            dialogue_context=dialogue_context, assistant_answer=rag_recsum_response
        )
        base_memory_bank_single_score = self.llm_scorer.evaluate_single(
            dialogue_context=dialogue_context,
            assistant_answer=base_memory_bank_response,
        )
        rag_memory_bank_single_score = self.llm_scorer.evaluate_single(
            dialogue_context=dialogue_context, assistant_answer=rag_memory_bank_response
        )
        full_sessions_baseline_single_score = self.llm_scorer.evaluate_single(
            dialogue_context=dialogue_context,
            assistant_answer=full_sessions_baseline_response,
        )
        last_session_baseline_single_score = self.llm_scorer.evaluate_single(
            dialogue_context=dialogue_context,
            assistant_answer=last_session_baseline_response,
        )

        self._single_eval_update(
            self.base_recsum_single_result, base_recsum_single_score
        )
        self._single_eval_update(self.rag_recsum_single_result, rag_recsum_single_score)
        self._single_eval_update(
            self.base_memory_bank_single_result, base_memory_bank_single_score
        )
        self._single_eval_update(
            self.rag_memory_bank_single_result, rag_memory_bank_single_score
        )
        self._single_eval_update(
            self.full_sessions_baseline_single_result,
            full_sessions_baseline_single_score,
        )
        self._single_eval_update(
            self.last_session_baseline_single_result, last_session_baseline_single_score
        )

        variants = [
            base_recsum_response,
            rag_recsum_response,
            base_memory_bank_response,
            rag_memory_bank_response,
            full_sessions_baseline_response,
            last_session_baseline_response,
        ]
        pairs = list(itertools.combinations(variants, 2))

        random.shuffle(pairs)

        for var1, var2 in pairs:
            pairwise_score = self.llm_scorer.evaluate_pairwise(
                dialogue_context=dialogue_context, first_answer=var1, second_answer=var2
            )

            mapping = {
                base_recsum_response: "base_recsum",
                rag_recsum_response: "rag_recsum",
                base_memory_bank_response: "base_memory_bank",
                rag_memory_bank_response: "rag_memory_bank",
                full_sessions_baseline_response: "full_baseline",
                last_session_baseline_response: "last_baseline",
            }

            alg1 = mapping[var1]
            alg2 = mapping[var2]

            for criterion in ["correctness", "clarity", "context_handling"]:
                result = getattr(pairwise_score, criterion)

                if result == ComparisonResult.OPTION_1_BETTER:
                    setattr(
                        self.pairwise_result,
                        alg1,
                        getattr(self.pairwise_result, alg1) + 1,
                    )
                elif result == ComparisonResult.OPTION_2_BETTER:
                    setattr(
                        self.pairwise_result,
                        alg2,
                        getattr(self.pairwise_result, alg2) + 1,
                    )
                elif result == ComparisonResult.DRAW:
                    setattr(
                        self.pairwise_result,
                        alg1,
                        getattr(self.pairwise_result, alg1) + 1,
                    )
                    setattr(
                        self.pairwise_result,
                        alg2,
                        getattr(self.pairwise_result, alg2) + 1,
                    )

        self.message_count += 1

    @staticmethod
    def _single_eval_update(result: SingleResult, score: SingleChatAgentResult) -> None:
        result.correctness.append(score.correctness_score)
        result.clarity.append(score.clarity_score)
        result.context_handling.append(score.context_handling_score)

    def print_results(self) -> None:
        def avg(lst: list[int]) -> float:
            return sum(lst) / len(lst) if lst else 0

        print("\n===Single Evaluation Results ===")
        print(
            f"{'Algorithm':<25} | {'Correctness':<12} | {'Clarity':<8} | {'Context':<8}"
        )
        print("-" * 60)
        print(
            f"{'Base Recsum':<25} | {avg(self.base_recsum_single_result.correctness):<12.2f} | "
            f"{avg(self.base_recsum_single_result.clarity):<8.2f} | "
            f"{avg(self.base_recsum_single_result.context_handling):<8.2f}"
        )
        print(
            f"{'RAG Recsum':<25} | {avg(self.rag_recsum_single_result.correctness):<12.2f} | "
            f"{avg(self.rag_recsum_single_result.clarity):<8.2f} | "
            f"{avg(self.rag_recsum_single_result.context_handling):<8.2f}"
        )
        print(
            f"{'Base MemoryBank':<25} | {avg(self.base_memory_bank_single_result.correctness):<12.2f} | "
            f"{avg(self.base_memory_bank_single_result.clarity):<8.2f} | "
            f"{avg(self.base_memory_bank_single_result.context_handling):<8.2f}"
        )
        print(
            f"{'RAG MemoryBank':<25} | {avg(self.rag_memory_bank_single_result.correctness):<12.2f} | "
            f"{avg(self.rag_memory_bank_single_result.clarity):<8.2f} | "
            f"{avg(self.rag_memory_bank_single_result.context_handling):<8.2f}"
        )
        print(
            f"{'Full Sessions Baseline':<25} |"
            f" {avg(self.full_sessions_baseline_single_result.correctness):<12.2f} | "
            f"{avg(self.full_sessions_baseline_single_result.clarity):<8.2f} | "
            f"{avg(self.full_sessions_baseline_single_result.context_handling):<8.2f}"
        )
        print(
            f"{'Last Session Baseline':<25} |"
            f" {avg(self.last_session_baseline_single_result.correctness):<12.2f} | "
            f"{avg(self.last_session_baseline_single_result.clarity):<8.2f} | "
            f"{avg(self.last_session_baseline_single_result.context_handling):<8.2f}"
        )

        print("\n===Pairwise Evaluation Results ===")
        print(f"{'Base Recsum':<25}: {self.pairwise_result.base_recsum}")
        print(f"{'RAG Recsum':<25}: {self.pairwise_result.rag_recsum}")
        print(f"{'Base MemoryBank':<25}: {self.pairwise_result.base_memory_bank}")
        print(f"{'RAG MemoryBank':<25}: {self.pairwise_result.rag_memory_bank}")
        print(f"{'Full Sessions Baseline':<25}: {self.pairwise_result.full_baseline}")
        print(f"{'Last Session Baseline':<25}: {self.pairwise_result.last_baseline}")

        print("\n===Token Usage and Cost ===")
        print(
            f"{'Algorithm':<25} | {'Prompt tokens':<15} | {'Completion tokens':<18} | {'Total cost ($)':<12}"
        )
        print("-" * 80)
        for name, algo in [
            ("Base Recsum", self.base_recsum),
            ("RAG Recsum", self.rag_recsum),
            ("Base MemoryBank", self.base_memory_bank),
            ("RAG MemoryBank", self.rag_memory_bank),
            ("Full Sessions Baseline", self.full_baseline),
            ("Last Session Baseline", self.last_baseline),
        ]:
            print(
                f"{name:<25} | {algo.prompt_tokens:<15} | {algo.completion_tokens:<18} | {algo.total_cost:<12.5f}"  # type: ignore
            )

        print("\n===Processed Messages ===")
        print(f"Total messages processed: {self.message_count}")


def main() -> None:
    metric_calculator = CalculateAgentChatResponseMetrics()

    print("Starting Agent Chat metrics calculation...")
    metric_calculator.calculate()

    print("Calculation completed. Results:")
    metric_calculator.print_results()


if __name__ == "__main__":
    main()
