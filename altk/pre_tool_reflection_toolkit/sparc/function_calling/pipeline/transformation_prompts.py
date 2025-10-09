from typing import Dict, Any, List
from copy import deepcopy

# ──────────────────────────────────────────────────────────────────────────────
# 1) extract_units
# ──────────────────────────────────────────────────────────────────────────────


SINGLE_PARAM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "user_units_or_format": {
            "type": ["string", "null"],
            "description": (
                "The canonical unit or format attached to `user_value`, normalized to lowercase, singular form, "
                "and using standard abbreviations. "
                "If this unit/format is composed of multiple units or formats write them all in a comma-separated list (e.g., 'second, millisecond', 'byte, kilobyte'). "
                "If none, return an empty string ''.\n"
                "Examples:\n"
                "  - Time: 'second', 'millisecond', 'hour', 'day'\n"
                "  - Data: 'byte', 'kilobyte', 'megabyte', 'gigabyte'\n"
                "  - Temperature: 'celsius', 'fahrenheit', 'kelvin'\n"
                "  - Length: 'meter', 'centimeter', 'inch', 'foot, inch'\n"
                "  - Volume: 'liter', 'milliliter'\n"
                "  - Weight: 'kilogram', 'gram', 'pound'\n"
                "  - Currency: 'usd', 'eur', 'jpy'\n"
                "  - Date formats: 'yyyy-mm-dd', 'month day, year', 'iso8601'\n"
                "  - Identifiers: 'uuid', 'hex'\n"
                "If the user_value has no unit/format, return ''."
            ),
        },
        "user_value": {
            "type": ["string", "null"],
            "description": (
                "The full and exact value provided by the user or from the conversation history (e.g. previous messages) for this parameter, always as a raw string (but still should be the value with these units/format: user_units_or_format).\n"
                "Collect the full value (if needed from multiple parts in the conversation) and return it as-is, with the full value, which is the value with these units/format: user_units_or_format.\n"
                "Examples:\n"
                "  - Time quantities: '30' (seconds), '2' (milliseconds), '1.5' (hours)\n"
                "  - Data sizes: '1000' (MB), '2' (GB), '512' (bytes)\n"
                "  - Temperatures: '25' (°C), '77' (°F)\n"
                "  - Dates: 'December 1st, 2024' (month day, year), '2024-06-20' (yyyy-mm-dd)\n"
                "  - Numbers: '0.75' (decimal), '42' (integer)\n"
                "  - Identifiers: '550e8400-e29b-41d4-a716-446655440000' (UUID)\n"
                "If the user did not mention this parameter, return `null`."
            ),
        },
        "spec_units_or_format": {
            "type": ["string", "null"],
            "description": (
                "The canonical unit or format defined or implied by the parameter's JSON Schema, "
                "normalized to lowercase and singular form, using the same conventions as `user_units_or_format`. "
                "If this unit/format is composed of multiple units or formats write them all in a comma-separated list (e.g., 'second, millisecond', 'byte, kilobyte'). "
                "Examples: 'second', 'byte', 'yyyy-mm-dd', 'uuid'.\n"
                "If the spec and user_value use the same unit/format, return exactly that same canonical string for both. "
                "If the schema specifies no unit/format, return ''."
            ),
        },
        "transformation_summary": {
            "type": ["string", "null"],
            "description": (
                "A brief summary of the transformation needed to convert `user_value` to the `spec_units_or_format`. "
                "This should be a human-readable description of the conversion logic, not the actual code."
                "Examples:\n"
                "  - 'Convert seconds to milliseconds by multiplying by 1000 - e.g., 30 seconds multiplied by 1000 to be in milliseconds.'\n"
                "  - 'Convert bytes to megabytes by dividing by 1024 - e.g., 2048 bytes divided by 1024 to be in megabytes.'\n"
                "  - 'Convert Celsius to Kelvin by adding 273.15 - e.g., 25°C is added 273.15 to be in Kelvin.'\n"
                "  - 'Convert date string from 'month day, year' to 'yyyy-mm-dd' format - e.g., 'December 1st, 2024' converted to 'yyyy-mm-dd' format.'\n"
                "  - 'Convert foots and inches to centimeters by multiplying by 30.48 - e.g., 5 feet 10 inches (5*30.48 + 10*2.54) to be in centimeters.'\n"
                "If no transformation is needed (i.e., `user_units_or_format` matches `spec_units_or_format`), return ''"
            ),
        },
    },
    "required": [
        "user_value",
        "user_units_or_format",
        "spec_units_or_format",
        "transformation_summary",
    ],
}


def build_multi_extract_units_schema(params: List[str]) -> Dict[str, Any]:
    """
    Construct a JSON Schema whose top-level properties are each parameter name.
    Each parameter maps to an object matching SINGLE_PARAM_SCHEMA.
    """
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": params.copy(),
    }
    for pname in params:
        schema["properties"][pname] = deepcopy(SINGLE_PARAM_SCHEMA)
    return schema


# -------------------------------------------------------------------
# 2) System prompt template for multi-parameter unit/format extraction
# -------------------------------------------------------------------
# We include a `{schema}` placeholder, which will be replaced at runtime
# with a JSON-dumped version of the schema built for the current params.
MULTI_EXTRACT_UNITS_SYSTEM: str = """\
You are an expert in natural language understanding and API specifications.
Given:
  1. A user context (natural-language instructions).
  2. A JSON Schema snippet that describes **all** parameters the tool expects.
  3. A list of all parameter names.

Your task:
  For each parameter name, identify:
    - The "user_units_or_format" explicitly or implicitly attached to that value.
      (If none, return an empty string `""`.)
    - The raw "user_value" mentioned in the user context or conversation history, as a string.
    - The "spec_units_or_format" defined or implied by the JSON Schema (type/description).
      (If none, return an empty string `""`.)
    - A brief "transformation_summary" describing how to convert `user_value` to `spec_units_or_format`.

Respond with exactly one JSON object whose keys are the parameter names,
and whose values are objects with "user_value", "user_units_or_format", and "spec_units_or_format".
The JSON must match this schema exactly:

{schema}
"""


# -------------------------------------------------------------------
# 3) User prompt template for multi-parameter unit extraction
# -------------------------------------------------------------------
# Use Python .format(...) placeholders for:
#   context        = The conversation/context string
#   full_spec      = JSON.dumps(...) of the combined JSON Schema snippet for all params
#   parameter_names = Comma-separated list of parameter names
MULTI_EXTRACT_UNITS_USER: str = """\

Examples (multi-parameter):

1) Context: [{{"role":"user", "content":"Change the interval to 30 seconds and set threshold to 0.75."}},
  {{"role":"assistant", "content":"{{"id":"tool_call_1","type":"function","function":{{"name":"set_interval_and_threshold","arguments":{{"interval":30,"threshold":0.75}}}}}}"}}]
   Full Spec:
   {{
     "name": "set_interval_and_threshold",
     "description": "Set the interval and threshold for monitoring.",
     "parameters": {{
      "type": "object",
      "properties": {{
        "interval": {{
          "type": "integer",
          "description": "Interval duration in seconds"
        }},
        "threshold": {{
          "type": "number",
          "description": "Threshold limit (0.0 to 1.0)"
        }}
      }},
      "required": ["interval", "threshold"]
    }}
   }}
   Parameter names: "interval, threshold"
   -> {{
        "interval": {{
          "user_units_or_format":"second",
          "user_value":"30",
          "spec_units_or_format":"second",
          "transformation_summary":""
        }},
        "threshold": {{
          "user_units_or_format":"",
          "user_value":"0.75",
          "spec_units_or_format":"",
          "transformation_summary":""
        }}
      }}

2) Context: [{{"role":"user", "content":"Download up to 2 GB of data and retry 5 times."}},
  {{"role":"assistant", "content":"{{"id":"tool_call_2","type":"function","function":{{"name":"download_data","arguments":{{"size":"2147483648","retries":5}}}}}}"}}]
   Full Spec:
   {{
     "name": "download_data",
     "description": "Download data with specified size and retry count.",
     "parameters": {{
      "type": "object",
      "properties": {{
        "size": {{
          "type": "string",
          "description": "Size limit in bytes"
        }},
        "retries": {{
          "type": "integer",
          "description": "Maximum retry count"
        }}
      }},
      "required": ["size", "retries"]
    }}
   }}
   Parameter names: "size, retries"
   -> {{
        "size": {{
          "user_units_or_format":"gigabyte",
          "user_value":"2",
          "spec_units_or_format":"byte",
          "transformation_summary":"Convert gigabytes to bytes by multiplying by 1024^3 - e.g., 2 GB multiplied by 1024^3 to be in bytes."
        }},
        "retries": {{
          "user_units_or_format":"",
          "user_value":"5",
          "spec_units_or_format":"",
          "transformation_summary":""
        }}
      }}

3) Context: [{{"role":"user", "content":"Set backup_date to December 1st, 2024 and limit to 100MB."}},
  {{"role":"assistant", "content":"{{\"id\":\"tool_call_3\",\"type\":\"function\",\"function\":{{\"name\":\"set_backup_parameters\",\"arguments\":{{\"backup_date\":\"2024-12-01\",\"limit\":\"104857600\"}}}}}}"}}]
   Full Spec:
   {{
    "name": "set_backup_parameters",
    "description": "Set parameters for backup operation.",
    "parameters": {{
     "type": "object",
     "properties": {{
       "backup_date": {{
         "type": "string",
         "format": "date",
         "description": "Date of backup in YYYY-MM-DD"
       }},
       "limit": {{
         "type": "string",
         "description": "File size cap (in bytes)"
       }}
     }},
     "required": ["backup_date", "limit"]
    }}
   }}
   Parameter names: "backup_date, limit"
   -> {{
        "backup_date": {{
          "user_units_or_format":"month day, year",
          "user_value":"December 1st, 2024",
          "spec_units_or_format":"yyyy-mm-dd",
          "transformation_summary":"Convert 'month day, year' format to 'yyyy-mm-dd' - e.g., 'December 1st, 2024' converted to 'yyyy-mm-dd' format."

        }},
        "limit": {{
          "user_units_or_format":"megabyte",
          "user_value":"100",
          "spec_units_or_format":"byte",
          "transformation_summary":"Convert megabytes to bytes by multiplying by 1024^2 - e.g., 100 MB multiplied by 1024^2 to be in bytes."
        }}
      }}

Context:
{context}

Full Spec (JSON Schema snippet for all parameters):
{full_spec}

Parameter names: {parameter_names}

Please return exactly one JSON object matching the schema defined in the system prompt.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 2) generate_transformation_code
# ──────────────────────────────────────────────────────────────────────────────

# System prompt for code generation
GENERATE_CODE_SYSTEM: str = """\
You are an expert Python engineer. Generate a self-contained Python module that converts between arbitrary units or formats. Your code must define exactly two functions:

1. transformation_code(input_value: str) -> <transformed_type>
   - **Purpose**: Convert a string in OLD_UNITS into its equivalent in TRANSFORMED_UNITS.
   - **Behavior**:
     - Parse the numeric or textual content from `input_value` (e.g. “10 ms”, “December 1st, 2011”).
     - Attach the OLD_UNITS and perform a conversion to TRANSFORMED_UNITS using standard Python libraries (e.g. `pint`, `datetime`/`dateutil`, or built-ins).
     - Return the result as the specified `<transformed_type>` (e.g. `int`, `float`, `str`, `list[float]`, etc.).
   - **Error Handling**: If parsing or conversion is unsupported, raise a `ValueError` with a clear message.

2. convert_example_str_transformed_to_transformed_type(transformed_value: str) -> <transformed_type>
   - **Purpose**: Parse a raw string in the example transformed format into the same `<transformed_type>`.
   - **Behavior**:
     - Strip any non-numeric or formatting characters as needed.
     - Return the parsed value.
   - **Error Handling**: If parsing fails, raise a `ValueError`.

You will be provided with the following information:
- TRANSFORMATION SUMMARY: A brief description of the transformation logic, e.g., "Convert a footer and inch value to centimeters by multiplying by 30.48 the foot value and by 2.54 the inch value, e.g., '5 feet 10 inches' calculated by 5*30.48 + 10*2.54 to be in centimeters."
- OLD UNITS: The units or format of the input value (e.g., "millisecond", "celsius", "yyyy-mm-dd").
- EXAMPLE FORMAT OF OLD VALUE: An example string in the OLD UNITS (e.g, "1000" (milliseconds), "25" (celsius), "December 1st, 2011" (month day, year)).
- TRANSFORMED UNITS: The units or format of the transformed value (e.g., "second", "kelvin", "unix timestamp").
- EXAMPLE FORMAT OF TRANSFORMED VALUE: An example string (may not be fully representative - therefore you should only take in account the units when implementing the transformation logic) in the TRANSFORMED UNITS (e.g., "10 s", "[298.15]", "1322697600").
- TRANSFORMED TYPE: The type of the transformed value (e.g., `int`, `float`, `str`, `list[float]`).

Your response must be a valid Python script that defines the two functions above, with no additional text or formatting.
The script should be self-contained and runnable in a standard Python environment without any external dependencies (except for standard libraries).
If the transformation is not supported or possible with the standard Python libraries, return an empty string in the generated_code field.
Respond with ONLY a JSON object matching this schema (no Markdown fences, no extra text):
{
  "generated_code": "<full python script>"
}"""


generated_code_example1 = (
    "from datetime import datetime, timezone\n"
    "import dateutil.parser\n\n"
    "def transformation_code(input_value: str) -> int:\n"
    '    """\n'
    "    Convert a date string with the format 'month day, year' to a unix timestamp.\n\n"
    "    Args:\n"
    "        input_value (str): The date string to convert.\n\n"
    "    Returns:\n"
    "        int: The unix timestamp representing the date.\n\n"
    "    Example:\n"
    "        >>> transformation_code('December 1st, 2011')\n"
    "        estimated output: 1322697600\n"
    '    """\n\n'
    "    # Parse the date string, dateutil.parser automatically handles 'st', 'nd', 'rd', 'th'\n"
    "    dt = dateutil.parser.parse(input_value)\n\n"
    "    # Ensure the datetime is treated as UTC\n"
    "    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)\n\n"
    "    # Convert to Unix timestamp\n"
    "    return int(dt.timestamp())\n\n"
    "def convert_example_str_transformed_to_transformed_type(transformed_value: str) -> int:\n"
    '    """\n'
    "    Convert a string representation of a unix timestamp to an integer.\n\n"
    "    Args:\n"
    "        transformed_value (str): The string representation of the unix timestamp.\n\n"
    "    Returns:\n"
    "        int: The unix timestamp as an integer.\n\n"
    "    Example:\n"
    "        >>> convert_example_str_transformed_to_transformed_type('1322697600')\n"
    "        1322697600\n"
    '    """\n\n'
    "    # Strip any whitespace and convert to integer\n"
    "    transformed_value = transformed_value.strip()\n"
    "    return int(transformed_value)\n"
)

transformation_eval_example1 = (
    """
### Example 1:

TRANSFORMATION SUMMARY: Convert a date string with the format 'month day, year' to a unix timestamp, e.g., 'December 1st, 2024' converted to a unix timestamp.
OLD UNITS: month day, year
EXAMPLE FORMAT OF OLD VALUE: 'December 1st, 2011'
TRANSFORMED UNITS: unix timestamp
EXAMPLE FORMAT OF TRANSFORMED VALUE: '1322697600'
TRANSFORMED TYPE: int

RESPONSE:
{{"""
    + '"generated_code": "'
    + generated_code_example1
    + '"'
    + """}}"""
)

generated_code_example2 = (
    "def transformation_code(input_value: str) -> float:\n"
    '    """\n'
    "    Convert a string in milliseconds to seconds.\n\n"
    "    Args:\n"
    "        input_value (str): The input value in milliseconds.\n\n"
    "    Returns:\n"
    "        float: The converted value in seconds.\n\n"
    "    Example:\n"
    "        >>> transformation_code('1000')\n"
    "        estimated output: 1.0\n"
    '    """\n\n'
    "    return float(input_value.strip()) / 1000\n\n"
    "def convert_example_str_transformed_to_transformed_type(transformed_value: str) -> float:\n"
    '    """\n'
    "    Convert a string representation of seconds to a float.\n\n"
    "    Args:\n"
    "        transformed_value (str): The string representation of the value in seconds.\n\n"
    "    Returns:\n"
    "        float: The converted value in seconds.\n\n"
    "    Example:\n"
    "        >>> convert_example_str_transformed_to_transformed_type('10')\n"
    "        10.0\n"
    '    """\n\n'
    "    return float(transformed_value.strip())\n"
)

transformation_eval_example2 = (
    """
### Example 2:

TRANSFORMATION SUMMARY: Convert a string in milliseconds to seconds, e.g., '1000' (milliseconds) divided by 1000 to be in seconds.
OLD UNITS: millisecond
EXAMPLE FORMAT OF OLD VALUE: '1000'
TRANSFORMED UNITS: second
EXAMPLE FORMAT OF TRANSFORMED VALUE: '10'
TRANSFORMED TYPE: float

RESPONSE:
{{"""
    + '"generated_code": "'
    + generated_code_example2
    + '"'
    + """}}"""
)

generated_code_example3 = (
    "def transformation_code(input_value: str) -> list[float]:\n"
    '    """\n'
    "    Convert a temperature string in Celsius to Kelvin.\n\n"
    "    Args:\n"
    "        input_value (str): The temperature in Celsius.\n\n"
    "    Returns:\n"
    "        list[float]: The converted temperature in Kelvin as a list.\n\n"
    "    Example:\n"
    "        >>> transformation_code('25')\n"
    "        estimated output: [298.15]\n"
    '    """\n\n'
    "    # Convert Celsius to Kelvin (K = C + 273.15)\n"
    "    kelvin_value = float(input_value) + 273.15\n"
    "    # Return as a list with one element\n"
    "    return [kelvin_value]\n\n"
    "def convert_example_str_transformed_to_transformed_type(transformed_value: str) -> list[float]:\n"
    '    """\n'
    "    Convert a string representation of a temperature in Kelvin to a list of floats.\n\n"
    "    Args:\n"
    "        transformed_value (str): The temperature in Kelvin as a string.\n\n"
    "    Returns:\n"
    "        list[float]: The converted temperature in Kelvin as a list.\n\n"
    "    Example:\n"
    "        >>> convert_example_str_transformed_to_transformed_type('[35]')\n"
    "        [35.0]\n"
    '    """\n\n'
    "    # Remove the brackets and convert to float\n"
    "    transformed_value = transformed_value.strip()[1:-1]\n"
    "    # Return as a list with one element\n"
    "    return [float(transformed_value)]\n"
)

transformation_eval_example3 = (
    """
### Example 3:

TRANSFORMATION SUMMARY: Convert a temperature string in Celsius to Kelvin, e.g., '25' (Celsius) added 273.15 to be in Kelvin.
OLD UNITS: celsius
EXAMPLE FORMAT OF OLD VALUE: '25'
TRANSFORMED UNITS: kelvin
EXAMPLE FORMAT OF TRANSFORMED VALUE: '[35]'
TRANSFORMED TYPE: list

RESPONSE:
{{"""
    + '"generated_code": "'
    + generated_code_example3
    + '"'
    + """}}"""
)

transformation_eval_example4 = """
### Unsupported Transformation Example:

TRANSFORMATION SUMMARY:
OLD UNITS: unit1
EXAMPLE FORMAT OF OLD VALUE: ABC
TRANSFORMED UNITS: unit2
EXAMPLE FORMAT OF TRANSFORMED VALUE: DEF
TRANSFORMED TYPE: str

RESPONSE:
{{"generated_code": ""}}"""


# User prompt template for code generation
# Use Python format-style placeholders:
#   transformation_eval_examples, old_value, old_units, transformed_value, transformed_units, transformed_type
GENERATE_CODE_USER: str = (
    f"""\
Few-shot examples for how to convert:

{transformation_eval_example1}

{transformation_eval_example2}

{transformation_eval_example3}

{transformation_eval_example4}

"""
    + """\

TASK:

TRANSFORMATION SUMMARY: {transformation_summary}
OLD UNITS: {old_units}
EXAMPLE FORMAT OF OLD VALUE: {old_value}
TRANSFORMED UNITS: {transformed_units}
EXAMPLE FORMAT OF TRANSFORMED VALUE: {transformed_value}
TRANSFORMED TYPE: {transformed_type}

RESPONSE:
"""
)

# JSON Schema dict for validation
GENERATE_CODE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "generated_code": {
            "type": "string",
            "description": "The generated Python code for the transformation. Should be a valid Python script without any Markdown formatting.",
        }
    },
    "required": ["generated_code"],
}
