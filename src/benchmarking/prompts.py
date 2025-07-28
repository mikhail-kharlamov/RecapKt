from langchain_core.prompts import PromptTemplate

BASELINE_PROMPT = PromptTemplate.from_template("""
You are an advanced AI language model capable of engaging in personality-based conversations.
Respond to the user based on the provided dialogue context. Craft a response that is natural and
conversational.
Dialog context: {context}
Query: {query}
The response to user is:""")

SINGLE_EVALUATION_PROMPT = PromptTemplate.from_template("""
You are an impartial judge. You will be shown a Conversation Context, Personality of Speakers and Assistant Response.
#Faithfulness: Please evaluate whether the Assistant's response accurately represents the information provided in the
 conversation context and persona, without introducing false or misleading information.
#Informativeness: Please evaluate whether the Assistant's response provides useful, relevant, and comprehensive
 information that adds value to the conversation and addresses the user's needs effectively.
#Coherency: Please evaluate whether the Assistant's response maintains a coherent and logical flow of conversation
 based on the evolving context. A response with good context coherence can understand and respond appropriately to
 changes in conversation topics, providing smooth and sensible interactions.
Conversation Context: {context}
Personality: {memory}
Assistant Response: {response}
Begin your evaluation by providing a short explanation, then you must rate the Assistant Response
 on an integer score of 1 (very bad) to 100 (very good).""")

PAIRWISE_EVALUATION_PROMPT = PromptTemplate.from_template("""
You are an expert evaluator tasked with comparing two AI assistant responses. You will evaluate them across three
 specific criteria and determine which response is better for each criterion.
## Input Information:
- Current Context: {context}
- Current Memory: {memory}
- Response 1: {first_response}
- Response 2: {second_response}

## Evaluation Criteria:

### 1. Faithfulness
Evaluate how well each response adheres to the given memory and maintains consistency with the established character
 traits. Consider:
- Does the response align with the assistant's defined personality?
- Are there any contradictions with previously established traits or information?
- Does the response maintain character consistency throughout?

### 2. Informativeness
Assess the quality and usefulness of information provided in each response. Consider:
- How comprehensive and relevant is the information?
- Does the response adequately address the user's query?
- Is the information accurate and helpful?
- Does it provide appropriate depth without being excessive?

### 3. Coherency
Evaluate the logical flow and contextual appropriateness of each response. Consider:
- Does the response flow logically and make sense in context?
- Are there smooth transitions between ideas?
- Does it appropriately respond to the conversation context?
- Is the response well-structured and easy to follow?

## Evaluation Options:
For each criterion, choose one of the following:
- Response 1 is better: Response 1 clearly outperforms Response 2
- Response 2 is better: Response 2 clearly outperforms Response 1
- Draw: Both responses are equally good

## Instructions:
Evaluate each criterion independently and provide your assessment for all three dimensions. Focus on objective
 evaluation based on the specific criteria rather than general preferences.
""")
