from src.summarize_algorithms.core.models import (
    BaseBlock,
    CodeBlock,
    Session,
    ToolCallBlock,
)
from src.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueSystem,
)


def main() -> None:
    role1 = "user"
    role2 = "assistant"

    sessions = [
        Session(
            [
                BaseBlock(
                    role1,
                    "Hi! Can you help me write a function to calculate the factorial?",
                ),
                CodeBlock(
                    role2,
                    "Of course! Here’s an example in Python:",
                    code="""
        def factorial(n):
            if n == 0 or n == 1:
                return 1
            return n * factorial(n - 1)
        """,
                ),
            ]
        ),
        Session(
            [
                BaseBlock(
                    role1, "I tested the function, but it crashes on large numbers."
                ),
                BaseBlock(
                    role2,
                    "Yes, for large numbers it’s better to use an iterative approach or math.factorial.",
                ),
                CodeBlock(
                    role2,
                    "Here’s the iterative version:",
                    code="""
        def factorial_iter(n):
            result = 1
            for i in range(2, n + 1):
                result *= i
            return result
        """,
                ),
            ]
        ),
        Session(
            [
                BaseBlock(role1, "Check if there are any errors in my sorting code."),
                CodeBlock(
                    role1,
                    "Here’s my code:",
                    code="""
        def sort_list(lst):
            for i in range(len(lst)):
                for j in range(len(lst) - 1):
                    if lst[j] > lst[j+1]:
                        lst[j], lst[j+1] = lst[j+1], lst[j]
        """,
                ),
                ToolCallBlock(
                    role=role2,
                    id="tool_1",
                    name="CodeAnalyzer",
                    arguments='{"code": "def sort_list..."}',
                    response="The code is correct, but the sorting is inefficient for large lists.",
                    content="Code check completed. The code is correct, but the sorting is inefficient for"
                    " large lists.",
                ),
            ]
        ),
        Session(
            [
                BaseBlock(role1, "How can it be improved?"),
                BaseBlock(role2, "You can use the built-in function sorted:"),
                CodeBlock(
                    role2,
                    "Example:",
                    code="""
        numbers = [5, 2, 9, 1]
        sorted_numbers = sorted(numbers)
        print(sorted_numbers)
        """,
                ),
            ]
        ),
        Session(
            [
                BaseBlock(role1, "Can you execute this code and show the result?"),
                CodeBlock(
                    role1,
                    "Code:",
                    code="""
        sum([i for i in range(1, 6)])
        """,
                ),
                ToolCallBlock(
                    role=role2,
                    id="tool_2",
                    name="PythonExecutor",
                    arguments='{"code": "sum([i for i in range(1, 6)])"}',
                    response="15",
                    content="Code execution result: 15",
                ),
            ]
        ),
    ]

    current_query = "Can you show how to sort in descending order?"

    system = MemoryBankDialogueSystem()

    result = system.process_dialogue(sessions, current_query)

    print(f"Prompts Tokens: {system.prompt_tokens}")
    print(f"Completion Tokens: {system.completion_tokens}")
    print(f"Total Cost: {system.total_cost}")
    print(f"Response: {result.response}")


if __name__ == "__main__":
    main()
