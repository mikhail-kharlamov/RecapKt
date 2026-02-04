import json

from pathlib import Path
from typing import Any

from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    Session,
    ToolCallBlock,
)


class Loader:
    """
    Utilities for loading tool schemas and dialogue sessions from exported JSON formats used in benchmark.

    The benchmark code supports multiple dataset formats ("data_type_1" and "data_type_2"). This class converts
    them into the project’s internal `Session`/`BaseBlock` representation.
    """

    @staticmethod
    def load_func_tools(path: Path | str) -> list[dict[str, Any]]:
        """
        Load tool/function specifications from a JSON file.

        :param path: path to JSON file.
        :return: list[dict[str, Any]]: tool specs.
        """
        with open(path, encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)

        return data

    @staticmethod
    def load_session_data_type_2(path: Path | str) -> Session:
        """
        Load a dialogue session from the "data_type_2" JSON export format.

        :param path: path to session JSON.
        :return: Session: parsed session in internal format.
        """
        result: list[BaseBlock] = []
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        i = 0
        while i < len(data):
            dict_block = data[i]
            block_type = dict_block["type"]

            if block_type in ("user", "system", "USER", "SYSTEM"):
                block = BaseBlock(
                    role=block_type.upper(),
                    content=dict_block["content"]
                )
                result.append(block)
                i += 1

            elif block_type in ("assistant", "ASSISTANT"):
                block = BaseBlock(
                    role=block_type.upper(),
                    content=dict_block["content"]
                )
                result.append(block)

                tool_calls = dict_block.get("toolCalls") or dict_block.get("tool_calls", [])
                if tool_calls:
                    i += 1
                    blocks = Loader.__process_tool_calls_data_type_2(
                        tool_calls,
                        data[i].get("toolResponses")
                        or data[i].get("tool_responses", [])
                    )
                    result.extend(blocks)

                i += 1

            elif block_type in ("tool_response", "TOOL_RESPONSE", "tool", "TOOL"):
                i += 1

            else:
                block = BaseBlock(
                    role=block_type.upper(),
                    content=str(dict_block.get("content", ""))
                )
                result.append(block)
                i += 1

        return Session(result)

    @staticmethod
    def load_session_data_type_1(path: Path | str) -> Session:
        """
        Load a dialogue session from the "data_type_1" JSON export format.

        :param path: path to session JSON.
        :return: Session: parsed session in internal format.
        """
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        chat = raw.get("serializableChat", {})
        messages = chat.get("messages", [])

        result: list[BaseBlock] = []

        for message in messages:
            message_type: str = message.get("type", "")

            if message_type.endswith("SerializableMessage.UserMessage"):
                prompt = message.get("prompt", "")
                block = BaseBlock(
                    role="USER",
                    content=prompt,
                )
                result.append(block)

            elif message_type.endswith("SerializableMessage.AssistantMessage"):
                response = message.get("response", "") or ""
                reasoning = message.get("reasoning", "") or ""
                if reasoning.strip():
                    content = f"{response}\n\n[reasoning]\n{reasoning}"
                else:
                    content = response

                block = BaseBlock(
                    role="ASSISTANT",
                    content=content,
                )
                result.append(block)

            elif message_type.endswith("SerializableMessage.ToolMessage"):
                block = Loader.__process_tool_calls_data_type_1(message)
                result.append(block)

            else:
                block = BaseBlock(
                    role="UNKNOWN",
                    content=str(message),
                )
                result.append(block)

        return Session(result)

    @staticmethod
    def __process_tool_calls_data_type_1(message: dict[str, Any]) -> ToolCallBlock:
        tool_call = message.get("toolCall") or {}
        tool_resp = message.get("toolResponse") or {}
        tool_id = tool_call.get("id", "")
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", "")
        response_result = tool_resp.get("result", "")
        if response_result == "failure":
            content = tool_resp.get("failure", "") or ""
        else:
            content = tool_resp.get("content", "") or ""
        block = ToolCallBlock(
            role="TOOL_RESPONSE",
            content=content,
            id=tool_id,
            name=name,
            arguments=arguments,
            response=response_result,
        )
        return block

    @staticmethod
    def __process_tool_calls_data_type_2(
            tool_calls: list[dict[str, Any]],
            tool_responses: list[dict[str, dict[str, Any]]]
    ) -> list[ToolCallBlock]:
        blocks: list[ToolCallBlock] = []
        for call in tool_calls:
            for tool_response in tool_responses:
                if "response" in tool_response:
                    response: dict[str, Any] | None = tool_response.get("response")
                elif "responseData" in tool_response:
                    response = json.loads(str(tool_response.get("responseData")))
                else:
                    continue

                if response is None:
                    continue

                if call["id"] == tool_response["id"]:
                    if response["result"] == "failure":
                        content = response["failure"]
                    else:
                        content = response["content"]

                    block = ToolCallBlock(
                        role="TOOL_RESPONSE",
                        content=content,
                        id=call["id"],
                        name=call["name"],
                        arguments=call["arguments"],
                        response=response["result"]
                    )
                    blocks.append(block)
        return blocks


if __name__ == "__main__":
    json_file_template: str = "*.json"
    path_data_type_1: Path = Path("/Users/mikhailkharlamov/Documents/.../data_type_1")
    for file in path_data_type_1.glob(json_file_template):
        print(file)
        session = Loader.load_session_data_type_1(file)
        print(len(session), "- session")
        tools = []
        for block in session:
            if isinstance(block, ToolCallBlock):
                tools.append(block)
        print(len(tools), "- tools")
        print(len(tools) / len(session))

    path_data_type_2: Path = Path("/Users/mikhailkharlamov/Documents/.../data_type_2")
    for file in path_data_type_2.glob(json_file_template):
        print(file)
        session = Loader.load_session_data_type_2(file)
        print(len(session), "- session")
        tools = []
        for block in session:
            if isinstance(block, ToolCallBlock):
                tools.append(block)
        print(len(tools), "- tools")
        print(len(tools) / len(session))
