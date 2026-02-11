import json
import os

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
tools_and_schemas_path = Path(os.getenv("TOOLS_AND_SCHEMAS_PATH"))

PLAN_SCHEMA: dict[str, Any] = json.loads((tools_and_schemas_path / "output_schema.json").read_text(encoding="utf-8"))
TOOLS: list[dict[str, Any]] = json.loads((tools_and_schemas_path / "tools.json").read_text(encoding="utf-8"))
