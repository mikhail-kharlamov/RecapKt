from src.algorithms.summarize_algorithms.core.base_summarizer import BaseSummarizer
from src.algorithms.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    MemoryBankDialogueState,
    MemoryDialogueState,
    RecsumDialogueState,
    ResponseContext,
    Session,
    UpdateState,
)
from src.algorithms.summarize_algorithms.core.response_generator import ResponseGenerator
from src.utils.system_prompt_builder import MemorySections


def update_memory_node(
    summarizer_instance: BaseSummarizer, state: MemoryDialogueState
) -> MemoryDialogueState:
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
    state: MemoryDialogueState
) -> MemoryDialogueState:
    if isinstance(state, RecsumDialogueState):
        text_memory = state.latest_memory
    elif isinstance(state, MemoryBankDialogueState):
        text_memory_blocks = state.text_memory_storage.find_similar(state.query)
        text_memory = str(Session(text_memory_blocks))
    else:
        raise TypeError(
            f"Unsupported status type for update_memory_node: {type(state)}"
        )

    code_memory: list[BaseBlock] = []
    if state.code_memory_storage is not None:
        code_memory = state.code_memory_storage.find_similar(state.query)

    tool_memory: list[BaseBlock] = []
    if state.tool_memory_storage is not None:
        tool_memory = state.tool_memory_storage.find_similar(state.query)

    memory_sections = MemorySections(
        recap=text_memory if isinstance(state, RecsumDialogueState) else None,
        memory_bank=text_memory if isinstance(state, MemoryBankDialogueState) else None,
        code_knowledge=str(Session(code_memory)) if len(code_memory) > 0 else None,
        tool_memory=str(Session(tool_memory)) if len(tool_memory) > 0 else None,
    )

    final_response: ResponseContext = response_generator_instance.generate_response(
        last_session=state.last_session,
        user_query=state.query,
        memory=memory_sections,
        memory_mode="memory",
    )

    state._response = final_response.response
    state.prepared_messages = final_response.prepared_history
    return state


def should_continue_memory_update(state: DialogueState) -> str:
    if state.current_session_index < len(state.dialogue_sessions):
        return UpdateState.CONTINUE_UPDATE.value
    else:
        return UpdateState.FINISH_UPDATE.value
