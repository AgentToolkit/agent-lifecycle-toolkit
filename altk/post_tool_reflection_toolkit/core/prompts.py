CODE_GENERATION_ZERO_SHOT_PROMPT = """
You will be given a JSON object which contains information. Your task is to extract and return <<task_prefix>>

Write a Python function that:
    Starts the function with "def ".
    Identifies the structure of the input data, ensuring it checks for relevant keys and data types.
    Processes the provided data.
    Iterates through the data to extract relevant information.
    Cleans numeric strings by removing non-numeric characters before converting them to integers.
    Performs proper checks to ensure a key exists and is not None before querying its value.
    Returns a dictionary containing only the requested keys.

Final Check:
    The function must be formatted in Python markdown for direct execution.
    No explanations, comments, or additional text should be included.
    Do not include any example usage data.

data = <<json_obj>>

Python Function:
""".strip()


CODE_GENERATION_ZERO_SHOT_PROMPT_WITH_RESPONSE_SCHEMA = """
You will be given a JSON object as data which is a response from a REST API containing information returned from the API call.
You will be given a JSON schema of the response from the REST API returned from the API call.
Your task is to extract and return information from the JSON object which follows the JSON schema and answers the user query: <<task_prefix>>

You need to write a Python function that:
    Starts the function with "def ".
    Takes only the entire api response as input and doesn't have any other input.
    Identifies the structure of the input data, ensuring it checks for relevant keys and data types.
    When comparing strings, it should always convert both sides of the comparison to lowercase.
    Processes the provided data.
    Iterates through the data to extract relevant information.
    Cleans numeric strings by removing non-numeric characters before converting them to integers.
    Performs proper checks to ensure a key exists and is not None before querying its value.
    Returns only the requested data as a string and no other extra information or words.
    Do not add any extra keys or terms to the output.

Final Check:
    The function must be formatted in Python markdown for direct execution.
    No explanations, comments, or additional text should be included.
    Do not include any example usage data.

The JSON schema of the object given as data is as follows: <<json_schema>>

data = <<json_obj>>

Python Function:

""".strip()


CODE_GENERATION_ZERO_SHOT_PROMPT_WITH_COMPACT_RESPONSE = """
You will be given a JSON object as data which is an example or compact version of the response from a REST API containing information returned from the API call.
You will be given a JSON schema of the response from the REST API returned from the API call.
Your task is to take reference and return information from the JSON object which follows the JSON schema and answers the user query: <<task_prefix>>

You need to write a Python function that:
    Starts the function with "def ".
    Takes only the entire api response as input and doesn't have any other input.
    Identifies the structure of the input data, ensuring it checks for relevant keys and data types.
    When comparing strings, it should always convert both sides of the comparison to lowercase. This is mandatory.
    Processes the provided data.
    Iterates through the data to extract relevant information.
    Cleans numeric strings by removing non-numeric characters before converting them to integers.
    Performs proper checks to ensure a key exists and is not None before querying its value.
    Returns only the requested data as a string and no other extra information or words.
    The data to be returned from the function should be in string.
    Do not add any extra keys or terms to the output.

Final Check:
    The function must be formatted in Python markdown for direct execution.
    There should be ```python in the beginning and the ending should be ```.
    No explanations, comments, or additional text should be included.
    Do not include any example usage data.

The JSON schema of the object given as data is as follows: <<json_schema>>

data = <<json_obj>>

Python Function:

""".strip()


PROMPT_GET_NL_QUERY = """

You will be given a JSON object as data which is a response from a REST API containing information returned from the API call.
You will be given a JSON schema of the response from the REST API returned from the API call.
You will be given a user query – the original request from the user.
You will be given LLM thought – the model’s internal reasoning or restatement of the request.

Your task is to output:

Create one clear question that is aligned with both the User Query and the LLM Thought.
The question must preserve every detail from the User Query and remain consistent with the LLM Thought.
It must not contradict any detail.
The goal is to produce a question that could be asked directly to retrieve the relevant answer from the provided data.
Do not include any explanation in the output question.

Input:
User Query= <<user_query>>
LLM Thought= <<llm_thought>>
data = <<json_obj>>
JSON schema = <<json_schema>>

""".strip()
