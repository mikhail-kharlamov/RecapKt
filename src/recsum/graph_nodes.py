from enum import Enum

from src.recsum.models import DialogueState
from src.recsum.response_generator import ResponseGenerator
from src.recsum.summarizer import Summarizer


class UpdateState(Enum):
    CONTINUE_UPDATE = "continue_update"
    FINISH_UPDATE = "finish_update"


def update_memory_node(summarizer: Summarizer, state: DialogueState) -> DialogueState:
    current_dialogue_session = state.dialogue_sessions[state.current_session_index]

    new_memory = summarizer.summarize(
        state.latest_memory, str(current_dialogue_session)
    )

    state.memory.append(new_memory)
    state.current_session_index += 1

    return state


def generate_response_node(
    response_generator: ResponseGenerator, state: DialogueState
) -> DialogueState:
    final_response = response_generator.generate_response(
        state.latest_memory, str(state.dialogue_sessions[-1]), state.query
    )

    state.response = final_response
    return state


def should_continue_memory_update(state: DialogueState) -> str:
    if state.current_session_index < len(state.dialogue_sessions) - 1:
        return UpdateState.CONTINUE_UPDATE.value
    else:
        return UpdateState.FINISH_UPDATE.value
