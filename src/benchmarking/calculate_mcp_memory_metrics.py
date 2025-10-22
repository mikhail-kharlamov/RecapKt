import random

from datetime import datetime
from typing import Any

from src.benchmarking.llm_evaluation import LLMMemoryEvaluation
from src.benchmarking.metric_calculator import (
    CalculateMCPMetrics,
    MCPResponseResults,
    MetricStats,
    RawLLMData,
    RawSemanticData,
    SystemResults,
)
from src.benchmarking.semantic_similarity import SemanticSimilarity
from src.summarize_algorithms.core.models import RecsumDialogueState
from src.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueState,
    MemoryBankDialogueSystem,
)


class CalculateMCPMemoryMetrics(CalculateMCPMetrics):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.memory_bank = MemoryBankDialogueSystem(max_session_id=4)

        self.semantic_scorer = SemanticSimilarity(use_tokenizer=False)
        self.llm_scorer = LLMMemoryEvaluation()

        self.session_count = 0

        self._memory_bank_semantic_data = RawSemanticData()
        self._memory_bank_llm_data = RawLLMData()

    @property
    def results(self) -> MCPResponseResults:
        if not self._is_calculated:
            self.calculate()
        return MCPResponseResults(
            metadata={
                "timestamp": datetime.now().isoformat(),
                "n_samples": self.n_samples,
                "message_count": self.session_count,
                "version": "1.0",
            },
            recsum_results=SystemResults(
                semantic_precision=MetricStats.from_values(
                    self._recsum_semantic_data.precision
                ),
                semantic_recall=MetricStats.from_values(
                    self._recsum_semantic_data.recall
                ),
                semantic_f1=MetricStats.from_values(self._recsum_semantic_data.f1),
                llm_faithfulness=MetricStats.from_values(
                    self._recsum_llm_data.faithfulness
                ),
                llm_informativeness=MetricStats.from_values(
                    self._recsum_llm_data.informativeness
                ),
                llm_coherency=MetricStats.from_values(self._recsum_llm_data.coherency),
            ),
            baseline_results=SystemResults(
                semantic_precision=MetricStats.from_values(
                    self._memory_bank_semantic_data.precision
                ),
                semantic_recall=MetricStats.from_values(
                    self._memory_bank_semantic_data.recall
                ),
                semantic_f1=MetricStats.from_values(self._memory_bank_semantic_data.f1),
                llm_faithfulness=MetricStats.from_values(
                    self._memory_bank_llm_data.faithfulness
                ),
                llm_informativeness=MetricStats.from_values(
                    self._memory_bank_llm_data.informativeness
                ),
                llm_coherency=MetricStats.from_values(
                    self._memory_bank_llm_data.coherency
                ),
            ),
            pairwise_results=self._pairwise_data,
        )

    def calculate(self) -> None:
        dialogues = self.dataset.sessions.copy()
        for i, dialogue in enumerate(dialogues):
            print(f"Processing dialogue {i + 1}/{len(dialogues)}")

            self._process_dialogue(dialogue, i)

        self._is_calculated = True

    def _process_dialogue(self, dialogue: list, dialogue_index: int) -> None:
        ideal_session_memory = self.dataset._memory[dialogue_index]

        self.recsum.process_dialogue(dialogue, "")
        self.memory_bank.process_dialogue(dialogue, "")

        recsum_state = self.recsum.state
        memory_bank_state = self.memory_bank.state

        if not isinstance(recsum_state, RecsumDialogueState):
            raise TypeError(
                f"Expected recsum_state to be of type RecsumDialogueState, "
                f"got {type(recsum_state)}"
            )

        if not isinstance(memory_bank_state, MemoryBankDialogueState):
            raise TypeError(
                f"Expected memory_bank_state to be of type MemoryBankDialogueSystem, "
                f"got {type(memory_bank_state)}"
            )

        for i in range(len(recsum_state.text_memory)):
            recsum_memory = recsum_state.text_memory[i]
            memory_bank_memory = (
                memory_bank_state.text_memory_storage.get_session_memory(i)
            )
            ideal_memory = ideal_session_memory[i].memory

            self._update_semantic_scores(
                recsum_memory, memory_bank_memory, ideal_memory
            )
            self._update_llm_single_scores(
                recsum_memory, memory_bank_memory, ideal_memory
            )
            self._update_llm_pairwise_scores(
                recsum_memory, memory_bank_memory, ideal_memory
            )

            self.session_count += 1

    def _update_semantic_scores(
        self,
        recsum_memory: list[str],
        memory_bank_memory: list[str],
        ideal_memory: list[str],
    ) -> None:
        recsum_score = self.semantic_scorer.compute_similarity(
            recsum_memory, ideal_memory
        )
        self._recsum_semantic_data.recall.append(recsum_score.recall)
        self._recsum_semantic_data.precision.append(recsum_score.precision)
        self._recsum_semantic_data.f1.append(recsum_score.f1)

        memory_bank_score = self.semantic_scorer.compute_similarity(
            memory_bank_memory, ideal_memory
        )
        self._memory_bank_semantic_data.recall.append(memory_bank_score.recall)
        self._memory_bank_semantic_data.precision.append(memory_bank_score.precision)
        self._memory_bank_semantic_data.f1.append(memory_bank_score.f1)

    def _update_llm_single_scores(
        self,
        recsum_memory: list[str],
        memory_bank_memory: list[str],
        ideal_memory: list[str],
    ) -> None:
        recsum_score = self.llm_scorer.evaluate_single(
            ideal_memory="\n".join(ideal_memory),
            memory="\n".join(recsum_memory),
        )
        self._recsum_llm_data.faithfulness.append(recsum_score.faithfulness_score)
        self._recsum_llm_data.informativeness.append(recsum_score.informativeness_score)
        self._recsum_llm_data.coherency.append(recsum_score.coherency_score)

        memory_bank_score = self.llm_scorer.evaluate_single(
            ideal_memory="\n".join(ideal_memory),
            memory="\n".join(memory_bank_memory),
        )
        self._memory_bank_llm_data.faithfulness.append(
            memory_bank_score.faithfulness_score
        )
        self._memory_bank_llm_data.informativeness.append(
            memory_bank_score.informativeness_score
        )
        self._memory_bank_llm_data.coherency.append(memory_bank_score.coherency_score)

    def _update_llm_pairwise_scores(
        self,
        recsum_memory: list[str],
        memory_bank_memory: list[str],
        ideal_memory: list[str],
    ) -> None:
        randomize_order = random.random() < 0.5

        if randomize_order:
            score = self.llm_scorer.evaluate_pairwise(
                ideal_memory="\n".join(ideal_memory),
                first_memory="\n".join(recsum_memory),
                second_memory="\n".join(memory_bank_memory),
            )
            self._update_pairwise_counts(score, recsum_first=True)
        else:
            score = self.llm_scorer.evaluate_pairwise(
                ideal_memory="\n".join(ideal_memory),
                first_memory="\n".join(memory_bank_memory),
                second_memory="\n".join(recsum_memory),
            )
            self._update_pairwise_counts(score, recsum_first=False)

    def print_results(self) -> None:
        print(f"\nProcessed {self.session_count} Session\n")

        results = self.results

        self._print_semantic_results(results)

        self._print_llm_single_results(results)
        self._print_llm_pairwise_results(results)


def main() -> None:
    metric_calculator = CalculateMCPMemoryMetrics(1)

    print("Starting MCP metrics calculation...")
    metric_calculator.calculate()

    print("Calculation completed. Results:")
    metric_calculator.print_results()

    saved_path = metric_calculator.save_results_to_json()
    print(f"\nResults have been saved to: {saved_path}")


if __name__ == "__main__":
    main()
