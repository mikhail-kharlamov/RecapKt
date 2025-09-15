from langchain_core.prompts import PromptTemplate

SESSION_SUMMARY_PROMPT = PromptTemplate.from_template(
    """
You are an expert AI assistant specialized in extracting and summarizing all personal,
contextual, and role-specific information from a single conversation session.

Input:
- Session Messages: a chronological list of messages exchanged between the user and the assistant.

Your task:
1. Read every message in Session Messages.
2. For the **User**, extract lasting personal or contextual details
 (e.g., location changes, projects, interests, goals, preferences).
3. For the **Assistant**, extract any self-descriptions, capabilities, or role-definitions it mentioned.
4. Omit purely transactional, technical, or ephemeral chit-chat that doesn’t contribute to lasting memory.
5. Synthesize all extracted information into a single, uninterrupted sequence of standalone sentences:
   - Each sentence should begin with either “The user…” or “The assistant…”.
   - Use present tense and describe each fact clearly.
   - Include up to 12 sentences total, mixing user and assistant details as they appear chronologically.
6. Do not separate into sections—output one continuous list of sentences.

Session Messages:
{session_messages}
"""
)
