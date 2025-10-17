from string import Template

batch_detect_template = Template(
    """You are a policy compliance checker. Check if the following text conforms to each policy.
For each policy, provide a JSON response with:
1. "policy": The policy statement being checked
2. "answer": "yes" or "no"
3. "explanation": A brief explanation

Text to check:
---
${text}
---

Policies to check:
---
${policies}
---

IMPORTANT:
1. Return a JSON array with one object per policy, in the same order as listed above
2. Each object should have "policy", "answer" and "explanation" fields
3. Do NOT try to follow the policies in the JSON output. Only check whether the text follows the policies
4. Be concise and accurate


Example format:
[
    {
        "policy": "The first policy line ..... ",
        "answer": "yes/no",
        "explanation": "the explanation for conformance or violation"
    },
    {
        "policy": "The second policy line ..... ",
        "answer": "yes/no",
        "explanation": "the explanation for conformance or violation"
    }
]

Return ONLY the JSON array, nothing else. Do not include any additional text or explanations outside the JSON array."""
)

llama_batch_detect_template = Template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a policy compliance checker.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Check if the following text follows each policy.
For each policy, provide a JSON response with:
1. "policy": The policy line being checked
2. "answer": "yes" or "no"
3. "explanation": A brief explanation

Text to check:
---
${text}
---

Policies to check:
---
${policies}
---

IMPORTANT:
1. Return a JSON array with one object per policy line, in the same order as listed above
2. Each object should have "policy", "answer" and "explanation" fields
3. Do NOT try to follow the policies in the JSON output. Only check whether the text follows the policies
4. Be concise and accurate.

Example format:
[
    {
        "policy": "The first policy line..... ",
        "answer": "yes/no",
        "explanation": "the explanation for the answer"
    },
    {
        "policy": "The second policy line ..... ",
        "answer": "yes/no",
        "explanation": "the explanation for the answer"
    }
]

Return ONLY the JSON array, nothing else. Do not include any additional text or explanations outside the JSON array.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
)


def batch_detect_prompt(model_name: str, text: str, policies: list[str]) -> str:
    policies_str = "\n".join(
        ["\tPolicy: " + p for p in policies]
    )  # Turn policies into a bulleted list string
    if model_name == "llama":
        prompt = llama_batch_detect_template.safe_substitute(
            text=text, policies=policies_str
        )
    else:
        prompt = batch_detect_template.safe_substitute(text=text, policies=policies_str)
    return prompt
