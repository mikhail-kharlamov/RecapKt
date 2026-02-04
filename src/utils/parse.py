from pathlib import Path

from src.algorithms.summarize_algorithms.core.models import ToolCallBlock
from src.benchmark.tool_plan_benchmarking.load_session import Loader
from src.benchmark.tool_plan_benchmarking.run import BASE_DATA_PATH, JSON_FILE_TEMPLATE

path_data_type_1: Path = Path(BASE_DATA_PATH)  / "data_type_1"
for file in path_data_type_1.glob(JSON_FILE_TEMPLATE):
    session = Loader.load_session_data_type_1(file)
    for message in session.messages:
        if isinstance(message, ToolCallBlock):
            if message.id == "c584b0f6-ee9b-4ad0-aa90-a337fb92c9b7":
                print(file)
                print(session.messages.index(message))

path_data_type_2: Path = Path(BASE_DATA_PATH) / "data_type_2"
for file in path_data_type_2.glob(JSON_FILE_TEMPLATE):
    session = Loader.load_session_data_type_2(file)
    for message in session.messages:
        if isinstance(message, ToolCallBlock):
            if message.id == "c584b0f6-ee9b-4ad0-aa90-a337fb92c9b7":
                print(file)
                print(session.messages.index(message))
