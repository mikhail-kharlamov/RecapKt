import abc

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable


class Summarizer(abc.ABC):
    @abc.abstractmethod
    def summarize(self, previous_memory: str, dialogue_context: str) -> str:
        pass


class RecursiveSummarizer(Summarizer):
    def __init__(self, llm: BaseChatModel, prompt_template: PromptTemplate) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.chain = self._build_chain()

    def _build_chain(self) -> Runnable[dict[str, Any], str]:
        return self.prompt_template | self.llm | StrOutputParser()

    def summarize(self, previous_memory: str, dialogue_context: str) -> str:
        try:
            response = self.chain.invoke(
                {
                    "previous_memory": previous_memory,
                    "dialogue_context": dialogue_context,
                }
            )
            return response
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e
