from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable


class BaseSummarizer(ABC):
    """
    Base class for summarizers used to update long-term dialogue memory.

    A summarizer wraps an LLM + prompt into a reusable LangChain runnable (`self.chain`). Concrete implementations
    define how to build the chain and which inputs they accept in `summarize()`.
    """

    def __init__(self, llm: BaseChatModel, prompt: PromptTemplate) -> None:
        self.llm = llm
        self.prompt = prompt
        self.chain = self._build_chain()

    @abstractmethod
    def _build_chain(self) -> Runnable[dict[str, Any], Any]:
        pass

    @abstractmethod
    def summarize(self, *args: Any, **kwargs: Any) -> Any:
        """
        Summarize/update memory.

        Concrete implementations define the accepted inputs (e.g. previous memory + dialogue context) and the
        returned memory representation.

        :param args: positional arguments required by concrete summarizers.
        :param kwargs: keyword arguments required by concrete summarizers.
        :return: Any: summarizer-specific memory output.
        """
        pass
