from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable


class ResponseGenerator:
    def __init__(self, llm: BaseChatModel, prompt_template: PromptTemplate) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.chain = self._build_chain()

    def _build_chain(self) -> Runnable:
        return self.prompt_template | self.llm | StrOutputParser()

    def generate_response(self, memory: str, dialogue_context: str, query: str) -> str:
        try:
            response = self.chain.invoke(
                {
                    "latest_memory": memory,
                    "current_dialogue_context": dialogue_context,
                    "query": query,
                }
            )
            return response
        except Exception as e:
            raise ConnectionError(f"API request failed: {str(e)}") from e
