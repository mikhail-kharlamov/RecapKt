import json

from pathlib import Path
from typing import Any


class ChatSessionCombiner:
    def __init__(
        self,
        file_list: list[str],
        output_file: str = "combined_chat_history_sessions.json",
    ) -> None:
        self.file_list = file_list
        self.output_file = output_file
        self.combined_data: list[dict[str, Any]] = []

    @staticmethod
    def _extract_session_id(file_name: str) -> str:
        return Path(file_name).stem

    @staticmethod
    def _load_chat_file(file_name: str) -> dict[str, Any] | None:
        try:
            with open(file_name, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File {file_name} not found")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON in the {file_name} file")
            return None
        except Exception as e:
            print(f"Unexpected error when processing the {file_name} file: {e}")
            return None

    def _create_session_entry(
        self, file_name: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        session_id = self._extract_session_id(file_name)
        return {"session_id": session_id, "messages": data}

    def process_files(self) -> None:
        self.combined_data = []

        for file_name in self.file_list:
            data = self._load_chat_file(file_name)

            if data is not None:
                session_entry = self._create_session_entry(file_name, data)
                self.combined_data.append(session_entry)
        self.save_combined_data()

    def save_combined_data(self) -> None:
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(self.combined_data, f, ensure_ascii=False, indent=4)
            print(f"Files have been successfully combined into {self.output_file}")
        except Exception as e:
            print(f"Error saving the {self.output_file} file: {e}")

    def get_session_count(self) -> int:
        return len(self.combined_data)

    def get_session_ids(self) -> list[str]:
        return [session["session_id"] for session in self.combined_data]


def main() -> None:
    file_list = [
        "chat-history.json",
        "chat-history3.json",
        "chat-history5.json",
        "chat-history6.json",
        "chat-history4.json",
    ]

    combiner = ChatSessionCombiner(file_list)

    combiner.process_files()

    print(f"Sessions processed: {combiner.get_session_count()}")
    print(f"ID sessions: {', '.join(combiner.get_session_ids())}")


if __name__ == "__main__":
    main()
