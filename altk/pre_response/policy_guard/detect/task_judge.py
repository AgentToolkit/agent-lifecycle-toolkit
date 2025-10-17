import json
from string import Template

from altk.core.toolkit import ComponentConfig


def create_adherence_check_report(results: list[dict]) -> dict:
    """Creates a report on adherence checks"""
    pre_task_adh = 0
    post_task_adh = 0
    report = {}
    if len(results) == 0:
        return report

    for i in range(len(results)):
        if results[i]["pre_task_adherence"]["score"] == "Yes":
            pre_task_adh += 1
        if results[i]["post_task_adherence"]["score"] == "Yes":
            post_task_adh += 1

    report = {
        "pre_task_adherence": pre_task_adh / len(results),
        "post_task_adherence": post_task_adh / len(results),
    }
    return report


class TaskJudge:
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.task_completion_prompt = Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader whose job is to determine if a response is a valid response to a query. Your score should be "Yes" or "No". If a response is ambiguous your score should be "Yes". The score should only be "No" if the response is definitely not a valid response.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>

        --- Here is the query:
        ${question}
        ---

        --- Here is the response to check:
        ${response}
        ---

        Output a JSON object like this:
        {
            "explanation": "a brief explanation for the score",
            "score": "Yes/No"
        }

        Return ONLY one JSON object, nothing else. Do not include any additional text or explanations outside the JSON object.<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>""")

    def check_task_completion(self, task: str, response: str):
        prompt = self.task_completion_prompt.safe_substitute(
            question=task, response=response
        )
        task_response = self.config.llm_client.generate(prompt)
        decoder = json.JSONDecoder()
        try:
            task_response = task_response.replace("json", "")
            task_response = task_response.strip().strip("`").strip()
            task_response, end = decoder.raw_decode(task_response)

        except Exception as e:
            print("Task judge failed: ", str(e))
            task_response = {
                "explanation": "",
                "score": None,
                "raw": task_response,
                "error": str(e),
            }

        return task_response
