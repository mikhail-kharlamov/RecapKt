from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable


class BaseSummarizer(ABC):
    def __init__(self, llm: BaseChatModel, prompt: PromptTemplate) -> None:
        self.llm = llm
        self.prompt = prompt
        self.chain = self._build_chain()

    @abstractmethod
    def _build_chain(self) -> Runnable[dict[str, Any], Any]:
        pass

    @abstractmethod
    def summarize(self, *args: Any, **kwargs: Any) -> Any:
        pass
