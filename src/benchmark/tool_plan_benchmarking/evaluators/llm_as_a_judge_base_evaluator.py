import os
from abc import abstractmethod
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.algorithms.summarize_algorithms.core.models import Session, DialogueState, BaseBlock, OpenAIModels
from src.benchmark.models.dtos import MetricState
from src.benchmark.tool_plan_benchmarking.evaluators.base_evaluator import BaseEvaluator


class LLMAsAJudgeBaseEvaluator(BaseEvaluator):
    def __init__(self, mode: str | None = None, llm: BaseChatModel | None = None) -> None:
        super().__init__(mode=mode)

        load_dotenv()

        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            self.llm = llm or ChatOpenAI(
                model=OpenAIModels.GPT_4_O_MINI.value,  # changed
                api_key=SecretStr(api_key))
        else:
            raise ValueError("OPENAI_API_KEY environment variable is not loaded")

        self.chain = self._build_chain()

    def _build_chain(self) -> RunnableSerializable[dict[str, str], Any]:
        return self.pairwise_eval_prompt | self.llm.with_structured_output(
            self._get_pairwise_result_model()
        )

    @abstractmethod
    def evaluate(
            self,
            sessions: list[Session],
            query: str,
            state: DialogueState,
            reference: list[BaseBlock] | None = None
    ) -> MetricState:
        """
        Returns eval score of llm's answer with for query.
        :param sessions: previous user's sessions (other chats or contexts).
        :param query: the last user's query which response is evaluating.
        :param state: history of interactions with model's answer of the query.
        :param reference: reference model's answer.
        :return: MetricState: dataclass with information about evaluation (metric's name and score).
        """
        ...
