import os

from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.benchmarking.baseline_logger import BaselineLogger
from src.benchmarking.prompts import BASELINE_PROMPT
from src.summarize_algorithms.core.models import OpenAIModels, Session


class DialogueBaseline:
    def __init__(self, system_name: str, llm: Optional[BaseChatModel] = None) -> None:
        load_dotenv()

        self.system_name = system_name

        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            self.llm = llm or ChatOpenAI(
                model=OpenAIModels.GPT_5_MINI.value,
                api_key=SecretStr(api_key)
            )
        else:
            raise ValueError("OPENAI_API_KEY environment variable is not loaded")

        self.prompt_template = BASELINE_PROMPT
        self.chain = self._build_chain()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0

        self.baseline_logger = BaselineLogger()

    def _build_chain(self) -> Runnable[dict[str, Any], str]:
        return self.prompt_template | self.llm | StrOutputParser()

    def process_dialogue(self, sessions: List[Session], query: str, iteration: int | None = None) -> str:
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

        if iteration is not None:
            self.baseline_logger.log_iteration(
                system_name=self.system_name,
                query=query,
                iteration=iteration,
                sessions=sessions
            )

        return result
