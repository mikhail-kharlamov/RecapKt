import os

import openai


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo") -> None:
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def send_request(self, prompt: str) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            return response.output_text
        except Exception as e:
            raise ConnectionError(f"API request failed: {str(e)}") from e


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY environment variable is not set")
        return

    client = OpenAIClient(api_key)

    prompt = "Write a stack implementation in python"

    print(client.send_request(prompt))


if __name__ == "__main__":
    main()
