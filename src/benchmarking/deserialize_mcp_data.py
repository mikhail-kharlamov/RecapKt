import random

from typing import Any

from datasets import load_dataset

from src.recsum.models import Message, Session


class MCPDataset:
    data_name = "nayohan/multi_session_chat"

    def __init__(
        self, n_samples: int, session_length: int = 3, shuffle: bool = True
    ) -> None:
        self.n_samples = n_samples
        self.session_length = min(session_length, 3)
        self.shuffle = shuffle
        self._sessions: list[list[Session]] = []
        self._memory: list[list[str]] = []
        self._is_initialized = False

    def _initialize_data(self) -> None:
        if self._is_initialized:
            return

        dataset = load_dataset(self.data_name, split="test")
        zero_sessions_idx = [
            i - 2
            for i, ex in enumerate(dataset)
            if ex["session_id"] == self.session_length
        ]
        if self.shuffle:
            selected_indices = random.sample(zero_sessions_idx, self.n_samples)
        else:
            selected_indices = zero_sessions_idx[: self.n_samples]

        for idx in selected_indices:
            self._process_dialogue(dataset[idx : idx + self.session_length])

        self._is_initialized = True

    def _process_dialogue(self, dialogue_data: dict[str, list[Any]]) -> None:
        memory = self._extract_memory(dialogue_data)
        self._memory.append(memory)

        sessions = self._extract_sessions(dialogue_data)
        self._sessions.append(sessions)

    def _extract_memory(self, dialogue_data: dict[str, list[Any]]) -> list[str]:
        sessions_memory = []

        persona1_data = dialogue_data.get("persona1", [])
        persona2_data = dialogue_data.get("persona2", [])

        for i in range(self.session_length):
            memory1 = "".join(persona1_data[i]) if i < len(persona1_data) else ""
            memory2 = "".join(persona2_data[i]) if i < len(persona2_data) else ""

            formatted_memory = f"persona1: {memory1}\npersona2: {memory2}"
            sessions_memory.append(formatted_memory)

        return sessions_memory

    @staticmethod
    def _extract_sessions(dialogue_data: dict[str, list[Any]]) -> list[Session]:
        sessions = []

        dialogue_sessions = dialogue_data.get("dialogue", [])
        speaker_sessions = dialogue_data.get("speaker", [])

        for dialogue_msgs, speakers in zip(dialogue_sessions, speaker_sessions):
            messages = [
                Message(role=speaker, message=message)
                for message, speaker in zip(dialogue_msgs, speakers)
            ]
            sessions.append(Session(messages))

        return sessions

    @property
    def sessions(self) -> list[list[Session]]:
        self._initialize_data()
        return self._sessions

    @property
    def memory(self) -> list[list[str]]:
        self._initialize_data()
        return self._memory

    def __len__(self) -> int:
        return self.n_samples
