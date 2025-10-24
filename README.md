# RecapKt
[![CI](https://github.com/ArtemKushnir/RecapKt/actions/workflows/ci.yaml/badge.svg)](https://github.com/ArtemKushnir/RecapKt/actions/workflows/ci.yaml)

Implementation of recursive summarization of dialogues from the article [Recursively Summarizing Enables Long-Term Dialogue
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
You must convert your dataset to the list[Session] format. All necessary dataclasses are described in **src/recsum/models.py**

The algorithm's response will be presented in **DialogueState** format, which is also described in **src/recsum/models.py**
```python
# src/main.py

from src.recsum.dialogue_system import DialogueSystem
from src.recsum.models import Message, Session

def main() -> None:
    sessions = ...
    result = system.process_dialogue(sessions, current_query)

    #example visualize results
    print(f"Memory: {result.latest_memory}")
    print(f"Memory length: {len(result.memory) if result.memory is not None else -1}")
    print(f"Response: {result.response}")
```

### Run main
<pre><code>python3 -m src.main</code></pre>

### Run main (uv)
<pre><code>uv run recupkt</code></pre>
