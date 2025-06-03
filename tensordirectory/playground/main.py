from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any, List

# Import the MCP server instance
from tensordirectory.mcp_interface import mcp_server
from pydantic import BaseModel, ValidationError

app = FastAPI()

# Paths
base_path = Path(__file__).parent
templates_path = base_path / "templates"
static_path = base_path / "static"

# Mount static files directory
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_playground_home():
    try:
        with open(templates_path / "index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Playground Home Page Not Found</h1><p>Create index.html in tensordirectory/playground/templates/</p>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.get("/api/tools", response_class=JSONResponse)
async def get_api_tools():
    """
    Returns a list of available MCP tools with basic schema information.
    """
    tools_info_list = []
    # Iterate over mcp_server.router.tools (list of Tool instances)
    for tool_handler in mcp_server.router.tools:
        tool_name = tool_handler.name
        description = tool_handler.description or tool_handler.fn.__doc__ or "No description available."

        parameters = []
        if hasattr(tool_handler, 'model') and tool_handler.model:
            schema = tool_handler.model.schema()
            if 'properties' in schema:
                for param_name, param_info in schema.get('properties', {}).items():
                    parameters.append({
                        "name": param_name,
                        "type": param_info.get("type", "any"),
                        "title": param_info.get("title", param_name.replace("_", " ").title()),
                        "required": param_name in schema.get("required", [])
                    })
        elif tool_name == "query_tensor_directory": # Special handling for query_tensor_directory if no model
             description = tool_handler.fn.__doc__ or "Query the tensor directory using natural language."
             parameters = [
                {"name": "prompt", "type": "string", "title": "Prompt", "required": True},
                {"name": "params", "type": "string", "title": "Params (JSON string, optional)", "required": False}
             ]

        tools_info_list.append({
            "name": tool_name,
            "description": description,
            "parameters": parameters
        })
    return JSONResponse(content=tools_info_list) # Return a list directly

@app.get("/api/tools/{tool_name}/schema", response_class=JSONResponse)
async def get_tool_schema(tool_name: str):
    """
    Returns the detailed JSON schema for a specific tool's arguments model.
    """
    # Access tool definition via mcp_server.router.tools_by_name
    tool_definition = mcp_server.router.tools_by_name.get(tool_name)
    if not tool_definition:
        raise HTTPException(status_code=404, detail="Tool not found")

    if hasattr(tool_definition, 'model') and tool_definition.model:
        return JSONResponse(content=tool_definition.model.schema())
    elif tool_name == "query_tensor_directory":
        # Manual schema for query_tensor_directory if it doesn't have a Pydantic model
        # for its main 'prompt' argument in the standard way.
        return JSONResponse(content={
            "title": "QueryTensorDirectoryArgs", # Matching the test expectation
            "type": "object",
            "properties": {
                "prompt": {"title": "Prompt", "type": "string"},
                "params": {"title": "Params", "type": "object", "additionalProperties": True, "description": "JSON object for additional parameters. Optional."}
            },
            "required": ["prompt"]
        })
    else:
        return JSONResponse(content={"message": "This tool does not have a Pydantic arguments model or a custom schema defined."})

class ExecuteToolRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any]

@app.post("/api/execute_tool")
async def execute_tool_endpoint(request_data: ExecuteToolRequest):
    tool_name = request_data.tool_name
    args = request_data.args

    # Access tool definition via mcp_server.router.tools_by_name
    if tool_name not in mcp_server.router.tools_by_name:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")

    tool_definition = mcp_server.router.tools_by_name[tool_name]

    try:
        result = await tool_definition.execute_tool(args)
        return {"success": True, "result": result}

    except ValidationError as ve:
        raise HTTPException(status_code=422, detail={"error_type": "Validation Error", "errors": ve.errors()})
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error executing tool '{tool_name}' with args {args}: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail={"error_type": str(type(e).__name__), "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tensordirectory.playground.main:app", host="0.0.0.0", port=8080, reload=True)
