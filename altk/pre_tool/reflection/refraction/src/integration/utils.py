from typing import List, Dict, Any

import json
import re


def extract_tool_calls(prompt: str) -> List[Dict[str, Any]]:
    call_match = re.search(
        pattern=r".*<tool_calls>(?P<tool_calls>.*)</tool_calls>.*",
        string=prompt,
        flags=re.DOTALL,
    )

    if call_match is not None:
        tool_calls_str = call_match.group("tool_calls")

        try:
            tool_calls: List[Dict[str, Any]] = json.loads(tool_calls_str)
            return tool_calls

        except json.JSONDecodeError:
            return []

    return []
