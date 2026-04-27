from typing import Any


def parse_response_properties(response: dict[str, Any]) -> dict[str, Any]:
    return response["properties"] if "properties" in response else response
