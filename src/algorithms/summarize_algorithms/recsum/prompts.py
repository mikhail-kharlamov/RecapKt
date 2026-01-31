from langchain_core.prompts import PromptTemplate

MEMORY_UPDATE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """You are an advanced AI assistant responsible for maintaining and updating a memory of personal details
(personality traits, background, preferences, etc.) for both the user and the assistant.
You will be given two inputs:
- Previous Memory: The existing memory of the user and the assistant.
- Dialogue Context: The new conversation between the user and the assistant.
Your goal is to produce an updated memory by incorporating any new or changed information from the dialogue context
into the previous memory.

Instructions:
1. Review Previous Memory:
   Identify the key personality information currently stored for the user and the assistant.
2. Extract New Information:
   Analyze the dialogue context to find any new personal details or changes regarding the user or the assistant
   (e.g., updated preferences, location, interests, background).
3. Update and Structure:
   Combine the old and new information into a clear, concise memory.
   Use separate sections for the user and the assistant.
   Present each person’s memory as bullet points.
4. Format:
   Structure the memory as shown in the example below.
   Do not exceed 20 sentences in the entire memory.

Example:
Previous Memory:
User:
- From Italy
- Enjoys cooking
Assistant:
- Friendly and helpful
- Enjoys learning about different cultures
Dialogue Context:
User: "Hi, I'm excited to tell you I recently moved to New York City for work,
 and I'm starting to learn Spanish on the side!"
Assistant: "That's great! I'd be happy to help you find Spanish classes.
 I'm an AI assistant designed to help with such queries."
Updated Memory:
User:
- From Italy
- Recently moved to New York City
- Enjoys cooking
- Learning Spanish
Assistant:
- Friendly and helpful AI assistant
- Enjoys learning about different cultures

Inputs:
Previous Memory: {previous_memory}
Session Context: {dialogue_context}

Output:
Updated Memory:
User:
- ...
Assistant:
- ..."""
)
