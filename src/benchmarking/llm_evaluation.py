from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.benchmarking.prompts import (
    PAIRWISE_EVALUATION_AGENT_RESPONSE,
    PAIRWISE_EVALUATION_MEMORY_PROMPT,
    PAIRWISE_EVALUATION_RESPONSE_PROMPT,
    SINGLE_EVALUATION_AGENT_RESPONSE,
    SINGLE_EVALUATION_MEMORY_PROMPT,
    SINGLE_EVALUATION_RESPONSE_PROMPT,
)
from src.summarize_algorithms.core.models import OpenAIModels


class ComparisonResult(Enum):
    OPTION_1_BETTER = "option 1 is better"
    OPTION_2_BETTER = "option 2 is better"
    DRAW = "draw"


class SingleResult(BaseModel):
    faithfulness_score: int = Field(
        ge=0, le=100, description="assessment by the criterion of Faithfulness"
    )
    informativeness_score: int = Field(
        ge=0, le=100, description="assessment by the criterion of Informativeness"
    )
    coherency_score: int = Field(
        ge=0, le=100, description="assessment by the criterion of Coherency"
    )


class PairwiseResult(BaseModel):
    faithfulness: ComparisonResult = Field(
        description="Comparison of options based on Faithfulness"
    )
    informativeness: ComparisonResult = Field(
        description="Comparison of options based on Informativeness"
    )
    coherency: ComparisonResult = Field(
        description="Comparison of options based on Coherency"
    )


class SingleChatAgentResult(BaseModel):
    correctness_score: int = Field(
        ge=0, le=100, description="assessment by the criterion of Correctness"
    )
    clarity_score: int = Field(
        ge=0, le=100, description="assessment by the criterion of Clarity"
    )
    context_handling_score: int = Field(
        ge=0, le=100, description="assessment by the criterion of Context Handling"
    )


class PairwiseChatAgentResult(BaseModel):
    correctness: ComparisonResult = Field(
        description="Comparison of options based on Correctness"
    )
    clarity: ComparisonResult = Field(
        description="Comparison of options based on Clarity"
    )
    context_handling: ComparisonResult = Field(
        description="Comparison of options based on Context Handling"
    )


SingleResultType = TypeVar("SingleResultType", bound=BaseModel)
PairwiseResultType = TypeVar("PairwiseResultType", bound=BaseModel)


class BaseLLMEvaluation(Generic[SingleResultType, PairwiseResultType], ABC):
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm or ChatOpenAI(model=OpenAIModels.GPT_4_1.value, temperature=0.0)
        self.single_eval_prompt = self._get_single_eval_prompt()
        self.pairwise_eval_prompt = self._get_pairwise_eval_prompt()
        self.single_eval_chain = self._build_single_eval_chain()
        self.pairwise_eval_chain = self._build_pairwise_eval_chain()

    @abstractmethod
    def _get_single_eval_prompt(self) -> PromptTemplate:
        pass

    @abstractmethod
    def _get_pairwise_eval_prompt(self) -> PromptTemplate:
        pass

    @abstractmethod
    def _get_single_result_model(self) -> type[SingleResultType]:
        pass

    @abstractmethod
    def _get_pairwise_result_model(self) -> type[PairwiseResultType]:
        pass

    def _build_single_eval_chain(self) -> RunnableSerializable[dict[str, str], Any]:
        return self.single_eval_prompt | self.llm.with_structured_output(
            self._get_single_result_model()
        )

    def _build_pairwise_eval_chain(self) -> RunnableSerializable[dict[str, str], Any]:
        return self.pairwise_eval_prompt | self.llm.with_structured_output(
            self._get_pairwise_result_model()
        )

    @staticmethod
    def _safe_invoke(chain: RunnableSerializable, params: dict[str, str]) -> Any:
        try:
            return chain.invoke(params)
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e


class LLMResponseEvaluation(BaseLLMEvaluation[SingleResult, PairwiseResult]):
    def _get_single_eval_prompt(self) -> PromptTemplate:
        return SINGLE_EVALUATION_RESPONSE_PROMPT

    def _get_pairwise_eval_prompt(self) -> PromptTemplate:
        return PAIRWISE_EVALUATION_RESPONSE_PROMPT

    def _get_single_result_model(self) -> type[SingleResult]:
        return SingleResult

    def _get_pairwise_result_model(self) -> type[PairwiseResult]:
        return PairwiseResult

    def evaluate_single(self, context: str, memory: str, response: str) -> SingleResult:
        params = {"context": context, "memory": memory, "response": response}
        return self._safe_invoke(self.single_eval_chain, params)

    def evaluate_pairwise(
        self, context: str, memory: str, first_response: str, second_response: str
    ) -> PairwiseResult:
        params = {
            "context": context,
            "memory": memory,
            "first_response": first_response,
            "second_response": second_response,
        }
        return self._safe_invoke(self.pairwise_eval_chain, params)


class LLMMemoryEvaluation(BaseLLMEvaluation[SingleResult, PairwiseResult]):
    def _get_single_eval_prompt(self) -> PromptTemplate:
        return SINGLE_EVALUATION_MEMORY_PROMPT

    def _get_pairwise_eval_prompt(self) -> PromptTemplate:
        return PAIRWISE_EVALUATION_MEMORY_PROMPT

    def _get_single_result_model(self) -> type[SingleResult]:
        return SingleResult

    def _get_pairwise_result_model(self) -> type[PairwiseResult]:
        return PairwiseResult

    def evaluate_single(self, ideal_memory: str, memory: str) -> SingleResult:
        params = {"generated_memory": memory, "ideal_memory": ideal_memory}
        return self._safe_invoke(self.single_eval_chain, params)

    def evaluate_pairwise(
        self, ideal_memory: str, first_memory: str, second_memory: str
    ) -> PairwiseResult:
        params = {
            "first_memory": first_memory,
            "second_memory": second_memory,
            "ideal_memory": ideal_memory,
        }
        return self._safe_invoke(self.pairwise_eval_chain, params)


class LLMChatAgentEvaluation(
    BaseLLMEvaluation[SingleChatAgentResult, PairwiseChatAgentResult]
):
    def _get_single_eval_prompt(self) -> PromptTemplate:
        return SINGLE_EVALUATION_AGENT_RESPONSE

    def _get_pairwise_eval_prompt(self) -> PromptTemplate:
        return PAIRWISE_EVALUATION_AGENT_RESPONSE

    def _get_single_result_model(self) -> type[SingleChatAgentResult]:
        return SingleChatAgentResult

    def _get_pairwise_result_model(self) -> type[PairwiseChatAgentResult]:
        return PairwiseChatAgentResult

    def evaluate_single(
        self, dialogue_context: str, assistant_answer: str
    ) -> SingleChatAgentResult:
        params = {
            "dialogue_context": dialogue_context,
            "assistant_answer": assistant_answer,
        }
        return self._safe_invoke(self.single_eval_chain, params)

    def evaluate_pairwise(
        self, dialogue_context: str, first_answer: str, second_answer: str
    ) -> PairwiseChatAgentResult:
        params = {
            "dialogue_context": dialogue_context,
            "first_answer": first_answer,
            "second_answer": second_answer,
        }
        return self._safe_invoke(self.pairwise_eval_chain, params)
