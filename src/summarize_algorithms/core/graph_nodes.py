from src.summarize_algorithms.core.base_summarizer import BaseSummarizer
from src.summarize_algorithms.core.models import (
    DialogueState,
    RecsumDialogueState,
    Session,
    UpdateState,
)
from src.summarize_algorithms.core.response_generator import ResponseGenerator


def update_memory_node(
    summarizer_instance: BaseSummarizer, state: DialogueState
) -> DialogueState:
    from src.summarize_algorithms.memory_bank.dialogue_system import (
        MemoryBankDialogueState,
    )

    current_dialogue_session = state.dialogue_sessions[state.current_session_index]

    if state.code_memory_storage is not None:
        code_blocks = current_dialogue_session.get_code_blocks()
        if len(code_blocks) > 0:
            state.code_memory_storage.add_memory(
                code_blocks, state.current_session_index
            )
    if state.tool_memory_storage is not None:
        tool_calls = current_dialogue_session.get_tool_calls()
        if len(tool_calls) > 0:
            state.tool_memory_storage.add_memory(
                tool_calls, state.current_session_index
            )

    text_blocks = current_dialogue_session.get_text_blocks()
    string_text_blocks = "\n".join([str(block) for block in text_blocks])

    if isinstance(state, RecsumDialogueState):
        new_memory = summarizer_instance.summarize(
            state.latest_memory, string_text_blocks
        )
        state.text_memory.append([memory.content for memory in new_memory])
    elif isinstance(state, MemoryBankDialogueState):
        new_memory = summarizer_instance.summarize(
            string_text_blocks, state.current_session_index
        )
        state.text_memory_storage.add_memory(new_memory, state.current_session_index)
    else:
        raise TypeError(
            f"Unsupported status type for update_memory_node: {type(state)}"
        )
    state.current_session_index += 1
    return state


def generate_response_node(
    response_generator_instance: ResponseGenerator,
    state: DialogueState
) -> DialogueState:
    from src.summarize_algorithms.memory_bank.dialogue_system import (
        MemoryBankDialogueState,
    )

    if isinstance(state, RecsumDialogueState):
        dialogue_memory = state.latest_memory
    elif isinstance(state, MemoryBankDialogueState):
        dialogue_memory = state.text_memory_storage.find_similar(state.query)
    else:
        raise TypeError(
            f"Unsupported status type for update_memory_node: {type(state)}"
        )

    if state.code_memory_storage is not None:
        code_memory = state.code_memory_storage.find_similar(state.query)
    else:
        code_memory = "Code Memory is missing"

    if state.tool_memory_storage is not None:
        tool_memory = state.tool_memory_storage.find_similar(state.query)
    else:
        tool_memory = "Tool Memory is missing"

    final_response = response_generator_instance.generate_response(
        text_memory=Session(dialogue_memory).to_langchain_messages(),
        code_memory=Session(code_memory).to_langchain_messages(),
        tool_memory=Session(tool_memory).to_langchain_messages(),
        query=state.query,
        sessions=[state.last_session]
    )

    state._response = final_response
    return state


def should_continue_memory_update(state: DialogueState) -> str:
    if state.current_session_index < len(state.dialogue_sessions):
        return UpdateState.CONTINUE_UPDATE.value
    else:
        return UpdateState.FINISH_UPDATE.value
