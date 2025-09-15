from typing import Any, List, Optional

from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.benchmarking.prompts import BASELINE_PROMPT
from src.summarize_algorithms.core.models import OpenAIModels, Session


class DialogueBaseline:
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm or ChatOpenAI(
            model=OpenAIModels.GPT_4_1_MINI.value, temperature=0.0
        )
        self.prompt_template = BASELINE_PROMPT
        self.chain = self._build_chain()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0

    def _build_chain(self) -> Runnable[dict[str, Any], str]:
        return self.prompt_template | self.llm | StrOutputParser()

    def process_dialogue(self, sessions: List[Session], query: str) -> str:
        context_messages = []
        for session in sessions:
            for message in session.messages:
                context_messages.append(f"{message.role}: {message.content}")
        context = "\n".join(context_messages)
        with get_openai_callback() as cb:
            result = self.chain.invoke({"context": context, "query": query})

            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens
            self.total_cost += cb.total_cost
        return result
