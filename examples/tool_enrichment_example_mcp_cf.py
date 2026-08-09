import json
from altk.core.toolkit import AgentPhase
from altk.core.llm import get_llm, GenerationMode
import os
from altk.build_time.tool_enrichment_toolkit.core.toolkit import (
    MCPCFToolEnrichBuildInput,
    MCPCFToolEnrichBuildOutput,
)
from altk.build_time.tool_enrichment_toolkit.utils.tool_enrichment import (
    MCPCFToolEnrichComponent,
)
from altk.build_time.tool_enrichment_toolkit.core.config import MCPCFToolEnrichConfig


def get_llm_client_obj(model_name="mistralai/mistral-small-3-1-24b-instruct-2503"):
    WatsonXAIClient = get_llm("watsonx")
    client = WatsonXAIClient(
        model_name=model_name,
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
    )
    return client


def example_tool_enrichment_with_toolkit(mcp_cf_tool_spec, config):
    tool_enrich_input = MCPCFToolEnrichBuildInput(mcp_cf_toolspec=mcp_cf_tool_spec)

    tool_enrich_middleware = MCPCFToolEnrichComponent()
    result = tool_enrich_middleware.process(
        data=tool_enrich_input, config=config, phase=AgentPhase.BUILDTIME
    )
    return result


mcp_cf_tool_spec = """  {
    "id": "01731d582ee148a39b0ff9b56bc26678",
    "originalName": "getV2Actions",
    "url": "http://localhost:9001/sse",
    "description": "from Salesloft",
    "requestType": "SSE",
    "integrationType": "MCP",
    "headers": null,
    "inputSchema": {
      "type": "object",
      "properties": {
        "ids": {
          "type": "string",
          "description": "Filter by action id",
          "example": 18,
          "x-ibm-examples": [
            345687678,
            767763892
          ]
        },
        "step_id": {
          "title": "Step name",
          "type": "string",
          "description": "Filter actions by step name",
          "x-ibm-category": "An-ID"
        },
        "type": {
          "title": "Type",
          "type": "string",
          "description": "Filter actions by type. Can be one of:email, phone,integration and other",
          "example": "filter",
          "x-ibm-enum-descriptions": [
            "phone",
            "email",
            "integration",
            "other"
          ],
          "x-ibm-examples": [
            "email",
            "phone"
          ]
        },
        "due_on[gt]": {
          "title": "Due on after the date",
          "type": "string",
          "description": "Returns actions that are due on after the provided date",
          "format": "date",
          "x-ibm-category": "Date-Time"
        },
        "due_on[gte]": {
          "title": "Due on greater than or equal to date",
          "type": "string",
          "description": "Returns actions that are greater than or equal to the provided date",
          "format": "date",
          "example": "2018-01-01T00:00:00Z",
          "x-ibm-category": "Date-Time",
          "x-ibm-examples": [
            "2018-02-01T00:00:00Z",
            "2018-03-01T00:00:00Z"
          ]
        },
        "due_on[lt]": {
          "title": "Due on before the date",
          "type": "string",
          "description": "Returns actions that are due on before the provided date",
          "format": "date",
          "example": "2018-01-01T00:00:00Z",
          "x-ibm-category": "Date-Time",
          "x-ibm-examples": [
            "2018-01-02T00:00:00Z",
            "2018-01-03T00:00:00Z"
          ]
        },
        "due_on[lte]": {
          "title": "Due on less than or equal to date",
          "type": "string",
          "description": "Returns actions that are less than or equal to the provided date",
          "format": "date",
          "example": "2018-01-01T00:00:00Z",
          "x-ibm-category": "Date-Time"
        },
        "user_guid": {
          "title": "User guid",
          "type": "string",
          "description": "Filters actions by the user's guid",
          "example": "12345678-1234-1234-1234-123456789012",
          "x-ibm-examples": [
            12345678,
            12355668
          ]
        },
        "person_id": {
          "title": "Person name",
          "type": "string",
          "description": "Filter actions by person name",
          "example": "123456",
          "x-ibm-category": "An-ID",
          "x-ibm-examples": [
            "319104110",
            "345678993"
          ]
        },
        "cadence_id": {
          "title": "Cadence name",
          "type": "string",
          "description": "Filter actions by cadence name",
          "example": "123456789",
          "x-ibm-category": "An-ID",
          "x-ibm-examples": [
            "1030231",
            "1034653"
          ]
        },
        "multitouch_group_id": {
          "title": "Multitouch group id",
          "type": "string",
          "description": "Filter actions by multitouch group id",
          "example": 987667893,
          "x-ibm-examples": [
            345696969,
            767088963
          ]
        },
        "updated_at[gt]": {
          "title": "Updated on after the date",
          "type": "string",
          "description": "Returns actions that are updated after the provided date",
          "format": "date",
          "example": "2017-01-01T00:00:00Z",
          "x-ibm-category": "Date-Time",
          "x-ibm-examples": [
            "2018-01-01T00:00:00Z",
            "2019-01-01T00:00:00Z"
          ]
        },
        "updated_at[gte]": {
          "title": "Updated at greater than or equal to the date",
          "type": "string",
          "description": "Returns actions that are greater than the provided date",
          "format": "date",
          "example": "2017-01-01T00:00:00Z",
          "x-ibm-category": "Date-Time",
          "x-ibm-examples": [
            "2018-01-01T00:00:00Z",
            "2019-01-01T00:00:00Z"
          ]
        },
        "updated_at[lt]": {
          "title": "Updated on before the date",
          "type": "string",
          "description": "Returns actions that are updated before the provided date",
          "format": "date",
          "example": "2017-01-01T00:00:00Z",
          "x-ibm-category": "Date-Time",
          "x-ibm-examples": [
            "2017-01-01T00:00:00.123Z",
            "2017-01-01T00:00:00.456Z"
          ]
        },
        "updated_at[lte]": {
          "title": "Updated at less than or equal to the date",
          "type": "string",
          "description": "Returns actions that are less than or equal to the provided date",
          "format": "date",
          "example": "2017-01-01T00:00:00Z",
          "x-ibm-category": "Date-Time",
          "x-ibm-examples": [
            "2017-01-01T00:00:00.123Z",
            "2017-01-01T00:00:00.456Z"
          ]
        },
        "sort_by": {
          "title": "Sort by",
          "enum": [
            "updated_at",
            "created_at"
          ],
          "type": "string",
          "description": "Key to sort on, must be one of: created_at, updated_at. Defaults to updated_at",
          "default": "updated_at",
          "example": "updated_at",
          "x-ibm-enum-descriptions": [
            "updated at",
            "created at"
          ],
          "x-ibm-examples": [
            "created_at"
          ]
        },
        "sort_direction": {
          "title": "Sort direction",
          "enum": [
            "ASC",
            "DESC"
          ],
          "type": "string",
          "description": "Direction to sort in, must be one of: ASC, DESC. Defaults to DESC",
          "default": "DESC",
          "example": "ASC",
          "x-ibm-enum-descriptions": [
            "ASC",
            "DESC"
          ],
          "x-ibm-examples": [
            "DESC"
          ]
        },
        "per_page": {
          "title": "Per page",
          "maximum": 100,
          "minimum": 1,
          "type": "integer",
          "description": "How many records to show per page in the range [1, 100]. Defaults to 25",
          "default": 25,
          "example": 1,
          "x-ibm-examples": [
            100,
            25
          ]
        },
        "page": {
          "title": "Page",
          "type": "integer",
          "default": 1,
          "example": 1,
          "x-ibm-examples": [
            2,
            3
          ],
          "description": "The current page to fetch results from. Defaults to 1."
        },
        "include_paging_counts": {
          "title": "Include paging counts",
          "type": "boolean",
          "default": false,
          "example": true,
          "x-ibm-enum-descriptions": [
            "True",
            "False"
          ],
          "x-ibm-examples": [
            false
          ],
          "description": "Whether to include total_pages and total_count in the metadata. Defaults to false"
        },
        "limit_paging_counts": {
          "title": "Limit paging counts",
          "type": "boolean",
          "example": true,
          "x-ibm-enum-descriptions": [
            "True",
            "False"
          ],
          "x-ibm-examples": [
            false
          ],
          "description": "Specifies whether the max limit of 10k records should be applied to pagination counts. Affects the total_count and total_pages data."
        }
      },
      "required": []
    },
    "annotations": {},
    "jsonpathFilter": "",
    "auth": null,
    "createdAt": "2025-10-16T06:51:21.209380",
    "updatedAt": "2025-10-16T06:51:21.209385",
    "enabled": true,
    "reachable": true,
    "gatewayId": "26c7544731ae4326a53a9e698c4e7119",
    "executionCount": 0,
    "metrics": {
      "totalExecutions": 0,
      "successfulExecutions": 0,
      "failedExecutions": 0,
      "failureRate": 0,
      "minResponseTime": null,
      "maxResponseTime": null,
      "avgResponseTime": null,
      "lastExecutionTime": null
    },
    "name": "salesloft-all-actions-getv2actions",
    "displayName": "Getv2Actions",
    "gatewaySlug": "salesloft-all-actions",
    "customName": "getV2Actions",
    "customNameSlug": "getv2actions",
    "tags": [],
    "createdBy": "admin@example.com",
    "createdFromIp": "127.0.0.1",
    "createdVia": "federation",
    "createdUserAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "modifiedBy": null,
    "modifiedFromIp": null,
    "modifiedVia": null,
    "modifiedUserAgent": null,
    "importBatchId": null,
    "federationSource": "salesloft all actions",
    "version": 1,
    "teamId": "a9a443a1e1f24619819a088e24d05e0b",
    "ownerEmail": "admin@example.com",
    "visibility": "public"
  }"""

config = MCPCFToolEnrichConfig(
    llm_client=get_llm_client_obj(
        model_name="mistralai/mistral-small-3-1-24b-instruct-2503"
    ),
    gen_mode=GenerationMode.TEXT,
)

result: MCPCFToolEnrichBuildOutput = example_tool_enrichment_with_toolkit(
    json.loads(mcp_cf_tool_spec), config
)
print(result.mcp_cf_toolspec)
