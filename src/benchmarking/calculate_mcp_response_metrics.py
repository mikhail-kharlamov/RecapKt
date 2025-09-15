import random

from datetime import datetime
from typing import Any

from src.benchmarking.baseline import DialogueBaseline
from src.benchmarking.llm_evaluation import LLMResponseEvaluation
from src.benchmarking.metric_calculator import (
    CalculateMCPMetrics,
    MCPResponseResults,
    MetricStats,
    RawLLMData,
    RawSemanticData,
    SystemResults,
)
from src.benchmarking.semantic_similarity import SemanticSimilarity


class CalculateMCPResponseMetrics(CalculateMCPMetrics):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.baseline = DialogueBaseline()
        self.semantic_scorer = SemanticSimilarity()
        self.llm_scorer = LLMResponseEvaluation()

        self.message_count = 0

        self._baseline_semantic_data = RawSemanticData()
        self._baseline_llm_data = RawLLMData()

    @property
    def results(self) -> MCPResponseResults:
        if not self._is_calculated:
            self.calculate()
        return MCPResponseResults(
            metadata={
                "timestamp": datetime.now().isoformat(),
                "n_samples": self.n_samples,
                "message_count": self.message_count,
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
                    self._baseline_semantic_data.precision
                ),
                semantic_recall=MetricStats.from_values(
                    self._baseline_semantic_data.recall
                ),
                semantic_f1=MetricStats.from_values(self._baseline_semantic_data.f1),
                llm_faithfulness=MetricStats.from_values(
                    self._baseline_llm_data.faithfulness
                ),
                llm_informativeness=MetricStats.from_values(
                    self._baseline_llm_data.informativeness
                ),
                llm_coherency=MetricStats.from_values(
                    self._baseline_llm_data.coherency
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
        ideal_response = dialogue[-1].messages.pop()

        while dialogue[-1].messages:
            self.message_count += 1
            query = dialogue[-1].messages.pop()

            recsum_response = self.recsum.process_dialogue(
                dialogue, query.content
            ).response
            baseline_response = self.baseline.process_dialogue(dialogue, query.content)

            self._update_semantic_scores(
                recsum_response, baseline_response, ideal_response.content
            )

            context = str(dialogue[-1])
            memory_session = self.dataset.memory[dialogue_index][-1].memory
            memory = "\n".join(memory_session)

            self._update_llm_single_scores(
                recsum_response, baseline_response, context, memory
            )
            self._update_llm_pairwise_scores(
                context, memory, recsum_response, baseline_response
            )

    def _update_semantic_scores(
        self, recsum_response: str, baseline_response: str, ideal_response: str
    ) -> None:
        recsum_score = self.semantic_scorer.compute_similarity(
            recsum_response, ideal_response
        )
        self._recsum_semantic_data.recall.append(recsum_score.recall)
        self._recsum_semantic_data.precision.append(recsum_score.precision)
        self._recsum_semantic_data.f1.append(recsum_score.f1)

        baseline_score = self.semantic_scorer.compute_similarity(
            baseline_response, ideal_response
        )
        self._baseline_semantic_data.recall.append(baseline_score.recall)
        self._baseline_semantic_data.precision.append(baseline_score.precision)
        self._baseline_semantic_data.f1.append(baseline_score.f1)

    def _update_llm_single_scores(
        self, recsum_response: str, baseline_response: str, context: str, memory: str
    ) -> None:
        recsum_score = self.llm_scorer.evaluate_single(
            context=context, memory=memory, response=recsum_response
        )
        self._recsum_llm_data.faithfulness.append(recsum_score.faithfulness_score)
        self._recsum_llm_data.informativeness.append(recsum_score.informativeness_score)
        self._recsum_llm_data.coherency.append(recsum_score.coherency_score)

        baseline_score = self.llm_scorer.evaluate_single(
            context=context, memory=memory, response=baseline_response
        )
        self._baseline_llm_data.faithfulness.append(baseline_score.faithfulness_score)
        self._baseline_llm_data.informativeness.append(
            baseline_score.informativeness_score
        )
        self._baseline_llm_data.coherency.append(baseline_score.coherency_score)

    def _update_llm_pairwise_scores(
        self, context: str, memory: str, recsum_response: str, baseline_response: str
    ) -> None:
        randomize_order = random.random() < 0.5

        if randomize_order:
            score = self.llm_scorer.evaluate_pairwise(
                context=context,
                memory=memory,
                first_response=recsum_response,
                second_response=baseline_response,
            )
            self._update_pairwise_counts(score, recsum_first=True)
        else:
            score = self.llm_scorer.evaluate_pairwise(
                context=context,
                memory=memory,
                first_response=baseline_response,
                second_response=recsum_response,
            )
            self._update_pairwise_counts(score, recsum_first=False)

    def print_results(self) -> None:
        print(f"\nProcessed {self.message_count} messages\n")

        results = self.results

        self._print_semantic_results(results)

        self._print_llm_single_results(results)
        self._print_llm_pairwise_results(results)


def main() -> None:
    metric_calculator = CalculateMCPResponseMetrics()

    print("Starting MCP metrics calculation...")
    metric_calculator.calculate()

    print("Calculation completed. Results:")
    metric_calculator.print_results()

    saved_path = metric_calculator.save_results_to_json()
    print(f"\nResults have been saved to: {saved_path}")


if __name__ == "__main__":
    main()
