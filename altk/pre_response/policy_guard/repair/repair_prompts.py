from string import Template
from altk.pre_response.policy_guard.core.utils import get_model_name

priority_repair_template = Template(
    """You are an editor who ensures that texts adhere to specific policies.

Here is the text to validate:
---
${text}
---

The text needs to be edited to bring it into adherence with the following policies:
${policies}

IMPORTANT INSTRUCTIONS:
1. Make ONLY the minimal changes necessary to fix the policy violations.
2. Focus ONLY on the specific parts of the text that violate the policies.
3. If relevant text snippets are provided, focus your changes on those specific sections.
4. Preserve the original tone, style, and intent of the text.
5. Do NOT rewrite the entire text - make surgical, targeted changes only.
6. If a policy has no relevant text snippets, make minimal changes to address the policy concern.
7. Address policies based on their priority level:
    - [HIGH PRIORITY]: Must be addressed first and given special attention
    - [MEDIUM PRIORITY]: Should be addressed after high priority items
    - [LOW PRIORITY]: Address these if they don't conflict with higher priority fixes
8. CRITICAL: When making changes, ensure you don't break compliance with the already conforming policies.
    The text already follows these policies correctly, so maintain their compliance while fixing violations.

Just return the repaired text with your minimal changes. Do not include any explanations, dashes or any other comments.

Output:
"""
)

ordered_repair_template = Template(
    """You are an editor who ensures that texts adhere to specific policies.

Here is the text to validate:
---
${text}
---

The text needs to be edited to bring it into adherence with the following policies:
${policies}

IMPORTANT INSTRUCTIONS:
1. Make ONLY the minimal changes necessary to fix the policy violations.
2. Focus ONLY on the specific parts of the text that violate the policies.
3. If relevant text snippets are provided, focus your changes on those specific sections.
4. Preserve the original tone, style, and intent of the text.
5. Do NOT rewrite the entire text - make surgical, targeted changes only.
6. If a policy has no relevant text snippets, make minimal changes to address the policy concern.
7. Address policies in the order in which they are presented.
8. CRITICAL: When making changes, ensure you don't break compliance with the already conforming policies.
    The text already follows these policies correctly, so maintain their compliance while fixing violations.

Just return the repaired text with your minimal changes. Do not include any explanations, dashes or any other comments.

Output:
"""
)


simple_repair_template = Template(
    """The following text violates one or more policies that it should have followed.

    <START_OF_TEXT>
    ${text}
    <END_OF_TEXT>

    POLICIES:
    ${policies}


    Rewrite the text so that it follows all policies while making only necessary changes and keeping the rest of the text unchanged.
    You may rewrite the text exactly once, and you must enclose the re-written text in <START_OF_REWRITE> <END_OF_REWRITE> tags.
"""
)

llama_simple_repair_template = Template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an editor and your task is to rewrite response.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>The following text violates one or more policies that it should have followed.

    <START_OF_TEXT>
    ${text}
    <END_OF_TEXT>

    POLICIES:
    ${policies}


    Rewrite the text so that it follows all policies while making only necessary changes and keeping the rest of the text unchanged.
    You may rewrite the text exactly once, and you must enclose the re-written text in <START_OF_REWRITE> <END_OF_REWRITE> tags.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
"""
)


simple_repair_single_template = Template(
    """The following text violates a policy that it should have followed.

    <START_OF_TEXT>
    ${text}
    <END_OF_TEXT>

    POLICY: ${policy}

    Rewrite the text so that it follows the policy while making only necessary changes and keeping the rest of the text unchanged.
    You may rewrite the text exactly once, and you must enclose the re-written text in <START_OF_REWRITE> <END_OF_REWRITE> tags.
"""
)

llama_simple_repair_single_template = Template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an editor and your task is to rewrite response.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>The following text violates a policy that it should have followed.

    <START_OF_TEXT>
    ${text}
    <END_OF_TEXT>

    POLICY: ${policy}

    Rewrite the text so that it follows the policy while making only necessary changes and keeping the rest of the text unchanged.
    You may rewrite the text exactly once, and you must enclose the re-written text in <START_OF_REWRITE> <END_OF_REWRITE> tags.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
"""
)

query_repair_single_template = Template(
    """You are an editor and your task is to rewrite response to ensure compliance with guidelines.
   The following text shows a query and a response. The response violates a guideline it should have followed.

    # Query
    ${query}

    # Response
    ${text}

    # Followed guidelines
    ${instr}

    # Violated guidelines
    ${policy}

    # Instructions
    Rewrite the response so that it follows the violated guideline while making only necessary changes and keeping the rest of the text unchanged. The rewrite must not break any of the listed followed guidelines that were already followed in the response.
    You may rewrite the text exactly once, and don't output anything other than the re-written response.
    You must enclose the re-written response in <START_OF_REWRITE> <END_OF_REWRITE> tags.
"""
)

allinone_repair_single_template = Template(
    """You are an editor and your task is to rewrite response to ensure compliance with guidelines.
   The following text shows a query, a response and a set of guidelines. The response may violate one or more of the guidelines it should have followed.

    # Query
    ${query}

    # Response
    ${text}

    # Guidelines
    ${instr}

    # Focus guideline
    ${policy}

    # Instructions
    Rewrite the response to make sure it follows the focus guideline while making only necessary changes and keeping the rest of the text unchanged. The rewrite must not break any of the other listed guidelines that were already followed in the response.
    You may rewrite the text exactly once, and don't output anything other than the re-written response.
    You must enclose the re-written response in <START_OF_REWRITE> <END_OF_REWRITE> tags.
"""
)


llama_query_repair_single_template = Template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an editor and your task is to rewrite a response to ensure compliance with guidelines.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>The following text shows a query and a response. The response violates a guideline it should have followed.

    <START_OF_QUERY>
    ${query}
    <END_OF_QUERY>

    <START_OF_RESPONSE>
    ${text}
    <END_OF_RESPONSE>

    <START_OF_FOLLOWED_GUIDELINES>
    ${instr}
    <END_OF_FOLLOWED_GUIDELINES>

    <START_OF_VIOLATED_GUIDELINE>
    ${policy}
    <END_OF_VIOLATED_GUIDELINE>

    Rewrite the response so that it follows the violated guideline while making only necessary changes and keeping the rest of the text unchanged. The rewrite must not break any of the listed followed guidelines that were already followed in the response.
    You may rewrite the text exactly once, and don't output anything other than the re-written response.
    You must enclose the re-written response in <START_OF_REWRITE> <END_OF_REWRITE> tags.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
"""
)

llama_allinone_repair_single_template = Template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an editor and your task is to rewrite a response to ensure compliance with guidelines.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>The following text shows a query, a response and a set of guidelines. The response may violate one or more of the guidelines it should have followed.

    <START_OF_QUERY>
    ${query}
    <END_OF_QUERY>

    <START_OF_RESPONSE>
    ${text}
    <END_OF_RESPONSE>

    <START_OF_GUIDELINES>
    ${instr}
    <END_OF_GUIDELINES>

    <START_OF_FOCUS_GUIDELINE>
    ${policy}
    <END_OF_FOCUS_GUIDELINE>

    Rewrite the response to make sure it follows the FOCUS GUIDELINE while making only necessary changes and keeping the rest of the text unchanged. The rewrite must not break any of the other listed guidelines that were already followed in the response.
    You may rewrite the text exactly once, and don't output anything other than the re-written response.
    You must enclose the re-written response in <START_OF_REWRITE> <END_OF_REWRITE> tags.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
"""
)

query_repair_template = Template(
    """You are an editor and your task is to rewrite a response to ensure compliance with guidelines.
    The following text shows a query with a set of guidelines and a response. The response violates one or more guidelines that it should have followed

    # Query
    ${query}
    The response must follow these guidelines:
    ${instr}

    # Response
    ${text}

    # Violated guidelines
    ${policies}

    # Instructions
    Rewrite the response so that it follows all violated guidelines while making only necessary changes and keeping the rest of the text unchanged. The rewrite must not break any of the guidelines that were already followed in the response.
    You may rewrite the text exactly once, and don't output anything other than the re-written response.
    You must enclose the re-written response in <START_OF_REWRITE> <END_OF_REWRITE> tags.
"""
)
query_gen_template = Template(
    """You are a helpful assistant and your task is to respond to queries. Your response must follow a set of guidelines.
    Here is a query with a set of guidelines that must be followed.

    # Query
    ${query}
    Your response must follow these guidelines:
    ${instr}

    # Instructions
    Write a response that follows all guidelines and don't output anything other than the response.
    You must enclose the response in <START_OF_RESPONSE> <END_OF_RESPONSE> tags.
"""
)

llama_query_gen_template = Template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant and your task is to respond to queries. Your response must follow a set of guidelines.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>Here is a query with a set of guidelines that must be followed.

    <START_OF_QUERY>
    ${query}
    Your response must follow these guidelines:
    ${instr}
    <END_OF_QUERY>

    Write a response that follows all guidelines and don't output anything other than the response.
    You must enclose the response in <START_OF_RESPONSE> <END_OF_RESPONSE> tags.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
"""
)

llama_query_repair_template = Template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an editor and your task is to rewrite response.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>The following text shows a query and a response. The response violates one or more guidelines from the query that it should have followed.

    <START_OF_QUERY>
    ${query}
    Your response must follow these guidelines:
    ${instr}
    <END_OF_QUERY>

    <START_OF_RESPONSE>
    ${text}
    <END_OF_RESPONSE>

    <START_OF_VIOLATED_GUIDELINES>
    ${policies}
    <END_OF_VIOLATED_GUIDELINES>

    Rewrite the response so that it follows all guidelines while making only necessary changes and keeping the rest of the text unchanged. The rewrite must not break any of the guidelines that were already followed in the response.
    You may rewrite the text exactly once, and don't output anything other than the re-written response.
    You must enclose the re-written response in <START_OF_REWRITE> <END_OF_REWRITE> tags.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
"""
)

all_in_one_repair_template = Template(
    """You are an editor and your task is to rewrite a response to ensure compliance with guidelines.
    The following text shows a query and a response. The response may violate one or more guidelines that it should have followed

    # Query
    ${query}

    # Response
    ${text}

    # Guidelines
    ${instr}

    # Instructions
    Rewrite the response so that it follows all guidelines while making only necessary changes and keeping the rest of the text unchanged. The rewrite must not break any of the guidelines that were already followed in the response.
    You may rewrite the text exactly once, and don't output anything other than the re-written response.
    You must enclose the re-written response in <START_OF_REWRITE> <END_OF_REWRITE> tags.
"""
)

llama_all_in_one_repair_template = Template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an editor and your task is to rewrite a response to ensure compliance with guidelines.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>The following text shows a query, a response and a set of guidelines. The response may violate one or more guidelines that it should have followed.

    <START_OF_QUERY>
    ${query}
    <END_OF_QUERY>

    <START_OF_RESPONSE>
    ${text}
    <END_OF_RESPONSE>

    <START_OF_GUIDELINES>
    ${instr}
    <END_OF_GUIDELINES>

    Rewrite the response so that it follows all guidelines while making only necessary changes and keeping the rest of the text unchanged. The rewrite must not break any of the guidelines that were already followed in the response.
    You may rewrite the text exactly once, and don't output anything other than the re-written response.
    You must enclose the re-written response in <START_OF_REWRITE> <END_OF_REWRITE> tags.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
"""
)

mapreduce_repair_template = Template(
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an editor and your task is to generate a response according to a user query and to ensure compliance with a set of guidelines.<|eot_id|>

<|start_header_id|>user<|end_header_id|>The following text shows the user query, a set of per-guideline responses, and the set of guidelines. Each per-guideline response is an example of a response that complies with the associated guideline.

<START_OF_QUERY>
${query}
<END_OF_QUERY>
${per_guideline_responses}
<START_OF_GUIDELINES>
${instr}
<END_OF_GUIDELINES>

Generate a response so that it follows all the guidelines. Use the per-guideline responses to help generate the final response that complies with all the guidelines.
Don't output anything other than the re-written response.
You must enclose the re-written response in <START_OF_REWRITE> <END_OF_REWRITE> tags.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
)

mapreduce_guideline_response_template = Template(
    """
<START_OF_PER_GUIDELINE_RESPONSE>
Guideline: ${guideline}

Response: ${response}
<END_OF_PER_GUIDELINE_RESPONSE>
"""
)


def simple_single_repair_prompt(model_id: str, text: str, policy: str) -> str:
    if get_model_name(model_id) == "llama":
        prompt = llama_simple_repair_single_template.safe_substitute(
            text=text, policy=policy
        )
    else:
        prompt = simple_repair_single_template.safe_substitute(text=text, policy=policy)
    return prompt


def simple_repair_prompt(model_id: str, text: str, policies: list[str]) -> str:
    policies_str = "\n".join(
        ["\t-" + p for p in policies]
    )  # Turn policies into a bulleted list string
    if get_model_name(model_id) == "llama":
        prompt = llama_simple_repair_template.safe_substitute(
            text=text, policies=policies_str
        )
    else:
        prompt = simple_repair_template.safe_substitute(
            text=text, policies=policies_str
        )
    return prompt


def query_single_repair_prompt(
    model_id: str, text: str, instructions: list[str], policy: str, query: str
) -> str:
    instr_str = "\n".join(["\t-" + p for p in instructions])
    if instr_str == "":
        instr_str = "none"

    if get_model_name(model_id) == "llama":
        prompt = llama_query_repair_single_template.safe_substitute(
            query=query, instr=instr_str, text=text, policy=policy
        )
    else:
        prompt = query_repair_single_template.safe_substitute(
            query=query, instr=instr_str, text=text, policy=policy
        )
    return prompt


def explanation_single_repair_prompt(
    model_id: str,
    text: str,
    instructions: list[str],
    violated_policy: str,
    query: str,
    explanation: str,
) -> str:
    instr_str = "\n".join(["\t-" + p for p in instructions])
    if instr_str == "":
        instr_str = "none"
    policy_str = violated_policy + " (Violation: " + explanation + ")"

    if get_model_name(model_id) == "llama":
        prompt = llama_query_repair_single_template.safe_substitute(
            query=query, instr=instr_str, text=text, policy=policy_str
        )
    else:
        prompt = query_repair_single_template.safe_substitute(
            query=query, instr=instr_str, text=text, policy=policy_str
        )
    return prompt


def allinone_single_repair_prompt(
    model_id: str, text: str, instructions: list[str], policy: str, query: str
) -> str:
    instr_str = "\n".join(["\t-" + p for p in instructions])
    if get_model_name(model_id) == "llama":
        prompt = llama_allinone_repair_single_template.safe_substitute(
            query=query, instr=instr_str, text=text, policy=policy
        )
    else:
        prompt = allinone_repair_single_template.safe_substitute(
            query=query, instr=instr_str, text=text, policy=policy
        )
    return prompt


def query_repair_prompt(
    model_id: str,
    text: str,
    followed_instr: list[str],
    violated_policies: list[str],
    query: str,
) -> str:
    violated_policies_str = "\n".join(
        ["\t-" + p for p in violated_policies]
    )  # Turn policies into a bulleted list string
    followed_instr_str = "\n".join(["\t-" + p for p in followed_instr])
    all_instr_str = followed_instr_str + "\n" + violated_policies_str

    if get_model_name(model_id) == "llama":
        prompt = llama_query_repair_template.safe_substitute(
            query=query,
            instr=followed_instr_str,
            text=text,
            policies=violated_policies_str,
        )
    else:
        prompt = query_repair_template.safe_substitute(
            query=query, instr=all_instr_str, text=text, policies=violated_policies_str
        )
    return prompt


def query_gen_prompt(model_id: str, instructions: list[str], query: str) -> str:
    instr_str = "\n".join(["\t-" + p for p in instructions])

    if get_model_name(model_id) == "llama":
        prompt = llama_query_gen_template.safe_substitute(query=query, instr=instr_str)
    else:
        prompt = query_gen_template.safe_substitute(query=query, instr=instr_str)
    return prompt


def allinone_repair_prompt(
    model_id: str, text: str, policies: list[str], query: str
) -> str:
    policies_str = "\n".join(
        ["\t-" + p for p in policies]
    )  # Turn policies into a bulleted list string
    if get_model_name(model_id) == "llama":
        prompt = llama_all_in_one_repair_template.safe_substitute(
            query=query, instr=policies_str, text=text
        )
    else:
        prompt = all_in_one_repair_template.safe_substitute(
            query=query, instr=policies_str, text=text
        )
    return prompt


def explanation_repair_prompt(
    model_id: str,
    text: str,
    followed_instr: list[str],
    violated_policies: list[str],
    query: str,
    explanations: list[str],
) -> str:
    violated_policies_str = "\n".join(
        ["\t-" + p for p in violated_policies]
    )  # Turn policies into a bulleted list string
    followed_instr_str = "\n".join(["\t-" + p for p in followed_instr])
    all_instr_str = followed_instr_str + "\n" + violated_policies_str
    # redo the violated policies string with explanations
    policies_str = "\n".join(
        [
            "\t-" + p + " (Violation: " + e + ")"
            for p, e in zip(violated_policies, explanations)
        ]
    )  # Turn policies into a bulleted list string

    if get_model_name(model_id) == "llama":
        prompt = llama_query_repair_template.safe_substitute(
            query=query, instr=followed_instr_str, text=text, policies=policies_str
        )
    else:
        prompt = query_repair_template.safe_substitute(
            query=query, instr=all_instr_str, text=text, policies=policies_str
        )
    return prompt


def priority_repair_prompt(
    text: str, policies: list[str], priorities: list[str]
) -> str:
    policies_str = ""
    for pol, pri in zip(policies, priorities):
        policies_str += "\t-" + pol + "  " + pri + "\n"
    prompt = priority_repair_template.safe_substitute(text=text, policies=policies_str)
    return prompt


def ordered_repair_prompt(
    text: str, policies: list[str], weights: list[float] = None, ranks: list[int] = None
) -> str:
    assert not (weights is None and ranks is None), (
        "One of weights or ranks must be set. "
    )
    if weights is not None:
        # Sort from highest to lowest weight
        sorted_policies = [x for _, x in sorted(zip(weights, policies), reverse=True)]
    elif ranks is not None:
        # Sort from lowest rank to highest rank
        sorted_policies = [x for _, x in sorted(zip(ranks, policies))]
    else:
        raise Exception("One of weights or ranks must be set. ")
    policies_str = "\n".join(
        ["\t-" + p for p in sorted_policies]
    )  # Turn policies into a bulleted list string
    prompt = ordered_repair_template.safe_substitute(text=text, policies=policies_str)
    return prompt


def mapreduce_repair_prompt(
    model_id: str, policies: list[str], responses: list[str], query: str
) -> str:
    policies_str = "\n".join(
        ["- " + p for p in policies]
    )  # Turn policies into a bulleted list string
    per_guideline_responses = ""
    for p, r in zip(policies, responses):
        per_guideline_responses += (
            mapreduce_guideline_response_template.safe_substitute(
                guideline=p, response=r
            )
        )
    prompt = mapreduce_repair_template.safe_substitute(
        query=query, instr=policies_str, per_guideline_responses=per_guideline_responses
    )
    return prompt
