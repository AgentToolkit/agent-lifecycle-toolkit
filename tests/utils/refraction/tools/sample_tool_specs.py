tools = [
    {
        "name": "w3",
        "description": "Get employee information from w3",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Employee email",
                }
            },
            "required": ["email"],
        },
        "output_parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "id",
                    "description": "Employee id",
                }
            },
        },
    },
    {
        "name": "author_workbench",
        "description": "Get employee publication information",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Employee id",
                }
            },
            "required": ["id"],
        },
        "output_parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "object",
                        "description": "Paper information",
                    }
                },
            },
        },
    },
    {
        "name": "hr_bot",
        "description": "Get employee information from HR",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Employee id",
                },
                "email": {
                    "type": "string",
                    "description": "Employee email",
                },
            },
        },
        "output_parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Employee id",
                },
                "info": {
                    "type": "object",
                    "description": "Employee information",
                },
            },
        },
    },
    {
        "name": "concur",
        "description": "Apply for travel approval",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_info": {
                    "type": "object",
                    "description": "Employee information",
                },
                "travel_justification": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paper": {
                                "type": "object",
                                "description": "Paper information",
                            }
                        },
                    },
                },
            },
            "required": ["employee_info", "travel_justification"],
        },
    },
]


tool_calls = [
    {
        "name": "w3",
        "arguments": {"email": "tchakra2@ibm.com"},
    },
    {
        "name": "author_workbench",
        "arguments": {"id": "$var1.id$"},
    },
    {
        "name": "hr_bot",
        "arguments": {"id": "$var1.id$", "email": "tchakra2@ibm.com"},
    },
    {
        "name": "concur",
        "arguments": {
            "employee_info": "$var3.info$",
            "travel_justification": "$var2.papers$",
        },
    },
]
