# RecapKt

[![CI](https://github.com/ArtemKushnir/RecapKt/actions/workflows/ci.yaml/badge.svg)](https://github.com/ArtemKushnir/RecapKt/actions/workflows/ci.yaml)

Implementation of recursive summarization of dialogues from the
article [Recursively Summarizing Enables Long-Term Dialogue
Memory in Large Language Models](https://arxiv.org/pdf/2308.15022)

## Technology Stack

- Python 3.8+
- LangChain
- LangGraph
- Tiktoken
- Numpy
- Sklearn

## Installation & Setup

### Clone Repository

<pre><code>git clone https://github.com/ArtemKushnir/RecapKt
cd RecapKt</code></pre>

### Install Dependencies (pip)

**Core dependencies**
<pre><code>pip install -r requirements.txt</code></pre>

**Development dependencies**
<pre><code>pip install -r requirements.dev.txt</code></pre>

### Install Dependencies (uv)

**Core dependencies**
<pre><code>uv pip install -r requirements.txt</code></pre>

**Development dependencies**
<pre><code>uv pip install -r requirements.dev.txt</code></pre>

## Quick Start

You must convert your dataset to the list[Session] format. All necessary dataclasses are described in *
*src/recsum/models.py**

The algorithm's response will be presented in **DialogueState** format, which is also described in *
*src/recsum/models.py**

```python
# src/main.py

from src.recsum.dialogue_system import DialogueSystem
from src.recsum.models import Message, Session


def main() -> None:
    sessions = ...
    result = system.process_dialogue(sessions, current_query)

    # example visualize results
    print(f"Memory: {result.latest_memory}")
    print(f"Memory length: {len(result.memory) if result.memory is not None else -1}")
    print(f"Response: {result.response}")
```

### Run main

<pre><code>python3 -m src.main</code></pre>

### Run main (uv)

<pre><code>uv run recupkt</code></pre>

### Metrics

| Model        | Method                 | Corr.     | Clarity   | Con. Hand. | Pairwise | Cost     |  
|--------------|------------------------|-----------|-----------|------------|----------|----------|
| GPT-4.1-mini | RecSum                 | **87.00** | 88.80     | **88.00**  | 42       | 0.03590$ | 
| GPT-4.1-mini | RagRecsum              | 86.60     | **89.20** | 86.20      | **48**   | 0.04013$ |
| GPT-4.1-mini | MemoryBank             | 80.20     | 81.60     | 73.40      | 18       | 0.02240$ |
| GPT-4.1-mini | RagMemoryBank          | 81.60     | 87.00     | 80.60      | 45       | 0.02805$ |
| GPT-4.1-mini | Full Sessions Baseline | 84.00     | 89.00     | 84.80      | 44       | 0.02699$ |
| GPT-4.1-mini | Last Session Baseline  | 84.60     | 88.00     | 82.80      | 35       | 0.01688$ |
|              |                        |           |           |            |          |          |
| GPT-4o       | RecSum                 | 78.80     | 80.80     | 78.40      | 42       | 0.23617$ | 
| GPT-4o       | RagRecsum              | 78.00     | 84.80     | 81.00      | 23       | 0.21423$ |
| GPT-4o       | MemoryBank             | 74.20     | 80.20     | 70.00      | 15       | 0.16974$ |
| GPT-4o       | RagMemoryBank          | **83.40** | 85.20     | 81.60      | 48       | 0.19809$ |
| GPT-4o       | Full Sessions Baseline | 83.00     | **88.20** | 86.40      | **60**   | 0.31348$ |
| GPT-4o       | Last Session Baseline  | 81.60     | 86.20     | **89.60**  | 44       | 0.10539$ |
|              |                        |           |           |            |          |          |
| GPT-5-nano   | RecSum                 | 87.60     | 88.80     | 89.00      | 35       | 0.02248$ | 
| GPT-5-nano   | RagRecsum              | 86.00     | 88.40     | 84.00      | 32       | 0.02207$ |
| GPT-5-nano   | MemoryBank             | 82.20     | 80.80     | 79.20      | 32       | 0.03594$ |
| GPT-5-nano   | RagMemoryBank          | 83.60     | 86.20     | 85.20      | 36       | 0.03288$ |
| GPT-5-nano   | Full Sessions Baseline | **88.00** | **90.40** | 91.40      | **49**   | 0.01153$ |
| GPT-5-nano   | Last Session Baseline  | 87.20     | **90.40** | **92.40**  | 42       | 0.00744$ |
|              |                        |           |           |            |          |          |
| GPT-5-mini   | RecSum                 | **89.00** | 89.80     | **91.60**  | 33       | 0.07664$ | 
| GPT-5-mini   | RagRecsum              | 83.60     | 87.60     | 84.20      | 34       | 0.07956$ |
| GPT-5-mini   | MemoryBank             | 84.60     | 87.20     | 83.60      | 30       | 0.09813$ |
| GPT-5-mini   | RagMemoryBank          | 85.20     | 87.40     | 86.60      | 45       | 0.10246$ |
| GPT-5-mini   | Full Sessions Baseline | 84.40     | **90.20** | 87.60      | 38       | 0.04678$ |
| GPT-5-mini   | Last Session Baseline  | 82.80     | 89.40     | 85.20      | **48**   | 0.02439$ |
