from enum import Enum


class MetricType(Enum):
    COHERENCE = "COHERENCE"
    F1_TOOL_STRICT = "F1_TOOL_STRICT"
    F1_TOOL = "F1_TOOL"


class AlgorithmName(Enum):
    BASE_RECSUM = "base_recsum"
    BASE_MEMORY_BANK = "base_memory_bank"
    RAG_RECSUM = "rag_recsum"
    RAG_MEMORY_BANK = "rag_memory_bank"
    FULL_BASELINE = "full_baseline"
    LAST_BASELINE = "last_baseline"
    SHORT_TOOLS = "short_tools"
    WEIGHTS = "weights"


class AlgorithmDirectory(Enum):
    BASE_RECSUM = "BaseRecsum"
    BASE_MEMORY_BANK = "BaseMemoryBank"
    RAG_RECSUM = "RagRecsum"
    RAG_MEMORY_BANK = "RagMemoryBank"
    FULL_BASELINE = "FullBaseline"
    LAST_BASELINE = "LastBaseline"
    SHORT_TOOLS = "ShortTools"
    WEIGHTS = "Weights"
