### REMOVE THIS?


def parse_result_for_json(result: str) -> str:
    # does various formatting to help with the json parsing

    def remove_from_start(string, substring):
        if string.startswith(substring):
            return string[len(substring) :].strip()
        return string.strip()

    def remove_from_end(string, substring):
        if string.endswith(substring):
            return string[: -len(substring)].strip()
        return string.strip()

    result = remove_from_end(result, "<|eom_id|>")

    result = remove_from_start(
        result, "Please provide the evaluation in the JSON format. ```json"
    )
    result = remove_from_start(result, "Please provide the JSON output. ```json")
    result = remove_from_start(
        result, "Please provide the evaluation in the structured JSON format."
    )
    result = remove_from_start(
        result, "Please provide the evaluation in the specified JSON format."
    )
    result = remove_from_start(
        result, "Please provide the evaluation in the JSON format."
    )

    result = remove_from_start(result, "```json")
    result = remove_from_end(result, "```")

    return result.strip()
