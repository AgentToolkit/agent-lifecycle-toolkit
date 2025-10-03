tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send an email to a given recipient with a subject and message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "The recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Body of the email message.",
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Query a knowledge base to retrieve relevant info on a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user question or search query.",
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "num_results": {
                                "type": "number",
                                "description": ("Number of top results to return."),
                            },
                            "domain_filter": {
                                "type": ["string", "null"],
                                "description": (
                                    "Optional domain to narrow the search (e.g."
                                    " 'finance', 'medical'). Pass null if not"
                                    " needed."
                                ),
                            },
                            "sort_by": {
                                "type": ["string", "null"],
                                "enum": [
                                    "relevance",
                                    "date",
                                    "popularity",
                                    "alphabetical",
                                ],
                                "description": (
                                    "How to sort results. Pass null if not needed."
                                ),
                            },
                        },
                        "required": ["num_results", "domain_filter", "sort_by"],
                        "additionalProperties": False,
                    },
                },
                "required": ["query", "options"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


tool_calls = [
    {
        "id": "call_12345xyz",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location":"Paris, France"}',
        },
    },
    {
        "id": "call_67890abc",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location":"Bogota, Colombia"}',
        },
    },
    {
        "id": "call_99999def",
        "type": "function",
        "function": {
            "name": "send_email",
            "arguments": '{"to":"bob@email.com","body":"Hi bob"}',
        },
    },
]

tools_internal = [item.get("function") for item in tools]
tool_calls_internal = [item.get("function") for item in tool_calls]
