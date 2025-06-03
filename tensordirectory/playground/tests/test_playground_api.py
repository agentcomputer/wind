import pytest
from fastapi.testclient import TestClient
from unittest import mock
from pydantic import ValidationError, BaseModel # Import BaseModel for dummy model in ValidationError

# FastAPI app instance from playground.main
from tensordirectory.playground.main import app
# mcp_server to inspect tools (though not strictly needed if mocking Tool.execute_tool)
from tensordirectory.mcp_interface import mcp_server
# The actual Tool class used by FastMCP server
# Path to Tool is mcp.server.fastmcp.tool.Tool for the OpenRouterAI mcp package
from mcp.server.fastmcp.tool import Tool as FastMCPTool


client = TestClient(app)

def test_get_playground_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "TensorDirectory API Playground" in response.text # Check for a key phrase

def test_get_api_tools():
    response = client.get("/api/tools")
    assert response.status_code == 200
    data = response.json()
    # The endpoint now returns a list directly
    assert isinstance(data, list)
    assert len(data) > 0

    tools_list = data # data is the list
    upload_tensor_tool_info = next((tool for tool in tools_list if tool["name"] == "upload_tensor"), None)
    assert upload_tensor_tool_info is not None
    assert "name" in upload_tensor_tool_info
    assert "description" in upload_tensor_tool_info
    assert "parameters" in upload_tensor_tool_info
    assert isinstance(upload_tensor_tool_info["parameters"], list)

    query_tool_info = next((tool for tool in tools_list if tool["name"] == "query_tensor_directory"), None)
    assert query_tool_info is not None


def test_get_tool_schema_success():
    tool_name = "upload_tensor"
    response = client.get(f"/api/tools/{tool_name}/schema")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert data["title"] == "TensorUploadArgs" # Check against the actual Pydantic model name
    assert "properties" in data
    assert "name" in data["properties"]
    assert "description" in data["properties"]
    assert "tensor_data" in data["properties"]

def test_get_tool_schema_query_tensor_directory():
    tool_name = "query_tensor_directory"
    response = client.get(f"/api/tools/{tool_name}/schema")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert data["title"] == "QueryTensorDirectoryArgs"
    assert "prompt" in data["properties"]
    assert data["properties"]["prompt"]["type"] == "string"

def test_get_tool_schema_not_found():
    tool_name = "non_existent_tool"
    response = client.get(f"/api/tools/{tool_name}/schema")
    assert response.status_code == 404
    assert "Tool not found" in response.json()["detail"]


# Using the correct path to FastMCP's Tool class method for mocking
@mock.patch.object(FastMCPTool, "execute_tool", new_callable=mock.AsyncMock)
async def test_execute_tool_success(mock_execute_tool_method):
    # Configure the mock to return a successful result (must be awaitable if original is async)
    mock_execute_tool_method.return_value = {"status": "success", "data": "mocked_tensor_uuid"}

    tool_name = "upload_tensor"
    # Verify this tool actually exists in the server to avoid false positives if mock is too broad
    assert tool_name in mcp_server.tools

    args_payload = {
        "name": "test_tensor",
        "description": "A test tensor",
        "tensor_data": [[1, 2], [3, 4]]
    }

    response = client.post("/api/execute_tool", json={"tool_name": tool_name, "args": args_payload})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"] == {"status": "success", "data": "mocked_tensor_uuid"}

    # Check that the mock was called on the correct instance with correct args
    # mcp_server.router.tools_by_name[tool_name] is the instance of the Tool class
    # .execute_tool is the method we mocked with new_callable=mock.AsyncMock
    mcp_server.router.tools_by_name[tool_name].execute_tool.assert_called_once_with(args_payload)


def test_execute_tool_not_found():
    response = client.post("/api/execute_tool", json={"tool_name": "fake_tool", "args": {}})
    assert response.status_code == 404
    assert "Tool 'fake_tool' not found" in response.json()["detail"]


# Dummy Pydantic model for ValidationError context
class DummyModel(BaseModel):
    name: str

@mock.patch.object(FastMCPTool, "execute_tool", new_callable=mock.AsyncMock)
async def test_execute_tool_validation_error(mock_execute_tool_method):
    validation_errors_dict = [{"loc": ("args", "name"), "msg": "field required", "type": "value_error.missing"}]
    # Create a ValidationError instance. It needs a model.
    mock_execute_tool_method.side_effect = ValidationError(errors=validation_errors_dict, model=DummyModel)

    tool_name = "upload_tensor"
    assert tool_name in mcp_server.tools
    args_payload = {"description": "missing name"}

    response = client.post("/api/execute_tool", json={"tool_name": tool_name, "args": args_payload})

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_type"] == "Validation Error"
    assert isinstance(data["detail"]["errors"], list)
    assert data["detail"]["errors"][0]["loc"] == ["args", "name"] # Check specific error detail

@mock.patch.object(FastMCPTool, "execute_tool", new_callable=mock.AsyncMock)
async def test_execute_tool_generic_error(mock_execute_tool_method):
    mock_execute_tool_method.side_effect = Exception("Something went very wrong")

    tool_name = "upload_tensor"
    assert tool_name in mcp_server.tools
    args_payload = {"name": "test", "description": "desc", "tensor_data": []}

    response = client.post("/api/execute_tool", json={"tool_name": tool_name, "args": args_payload})

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_type"] == "Exception"
    assert data["detail"]["message"] == "Something went very wrong"

# Test for a tool that doesn't use a Pydantic model for args (e.g. query_tensor_directory)
# if its handling in execute_tool is different.
# The current mock of execute_tool is general.
# If query_tensor_directory is called via execute_tool, the current tests should cover its success/error paths generically.

# Test for HTML home page
def test_get_html_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "Welcome to the TensorDirectory API Playground!" in response.text

# Test for static CSS file (if served by FastAPI, which it is)
def test_get_static_css():
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/css; charset=utf-8" # FastAPI should set this
    assert "body {" in response.text # Check for some known content

# Test for static JS file
def test_get_static_js():
    response = client.get("/static/js/main.js")
    assert response.status_code == 200
    # FastAPI might return application/javascript or text/javascript
    assert "javascript" in response.headers["content-type"]
    assert "DOMContentLoaded" in response.text
