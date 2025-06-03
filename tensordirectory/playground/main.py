from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any, List

# Import the MCP server instance
from tensordirectory.mcp_interface import mcp_server # mcp_server is an instance of FastMCP
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

    # Check if mcp_server has tool_definitions as per instruction for iteration
    if not hasattr(mcp_server, 'tool_definitions') or not isinstance(mcp_server.tool_definitions, list):
        # This indicates the attribute is not what's expected.
        # For the OpenRouterAI mcp package, tools are in mcp_server.tools (dict)
        # or mcp_server.router.tools (list via router)
        # However, strictly following the prompt to use mcp_server.tool_definitions:
        print("Error: mcp_server.tool_definitions is not a list or does not exist. Using mcp_server.tools.values() as fallback for now if available, or router.")
        source_iterable = []
        if hasattr(mcp_server, 'tools') and isinstance(mcp_server.tools, dict):
            source_iterable = mcp_server.tools.values()
        elif hasattr(mcp_server, 'router') and hasattr(mcp_server.router, 'tools') and isinstance(mcp_server.router.tools, list):
             source_iterable = mcp_server.router.tools
        else:
            raise HTTPException(status_code=500, detail="Server configuration error: Cannot find tool definitions.")
    else:
        source_iterable = mcp_server.tool_definitions

    for tool_handler in source_iterable:
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
        elif tool_name == "query_tensor_directory":
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
    return JSONResponse(content=tools_info_list)

@app.get("/api/tools/{tool_name}/schema", response_class=JSONResponse)
async def get_tool_schema(tool_name: str):
    """
    Returns the detailed JSON schema for a specific tool's arguments model.
    """
    # Access tool definition via mcp_server.tools_by_name
    if not hasattr(mcp_server, 'tools_by_name') or not isinstance(mcp_server.tools_by_name, dict):
        # Fallback if the direct attribute doesn't exist, though the prompt implies it should.
        print("Error: mcp_server.tools_by_name is not a dict or does not exist. Using mcp_server.tools as fallback if available, or router.")
        if hasattr(mcp_server, 'tools') and isinstance(mcp_server.tools, dict):
            tool_definition = mcp_server.tools.get(tool_name)
        elif hasattr(mcp_server, 'router') and hasattr(mcp_server.router, 'tools_by_name') and isinstance(mcp_server.router.tools_by_name, dict):
            tool_definition = mcp_server.router.tools_by_name.get(tool_name)
        else:
            raise HTTPException(status_code=500, detail="Server configuration error: Cannot find tool lookups.")
    else:
        tool_definition = mcp_server.tools_by_name.get(tool_name)

    if not tool_definition:
        raise HTTPException(status_code=404, detail="Tool not found")

    if hasattr(tool_definition, 'model') and tool_definition.model:
        return JSONResponse(content=tool_definition.model.schema())
    elif tool_name == "query_tensor_directory":
        return JSONResponse(content={
            "title": "QueryTensorDirectoryArgs",
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

    tool_definition = None
    # Access tool definition via mcp_server.tools_by_name
    if not hasattr(mcp_server, 'tools_by_name') or not isinstance(mcp_server.tools_by_name, dict):
        print("Error: mcp_server.tools_by_name is not a dict or does not exist. Using mcp_server.tools as fallback if available, or router.")
        if hasattr(mcp_server, 'tools') and isinstance(mcp_server.tools, dict): # Fallback
            tool_definition = mcp_server.tools.get(tool_name)
        elif hasattr(mcp_server, 'router') and hasattr(mcp_server.router, 'tools_by_name') and isinstance(mcp_server.router.tools_by_name, dict):
            tool_definition = mcp_server.router.tools_by_name.get(tool_name)
        else:
            raise HTTPException(status_code=500, detail="Server configuration error: Cannot find tool lookups for execution.")
    else:
         if tool_name not in mcp_server.tools_by_name: # Check existence
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found via tools_by_name.")
         tool_definition = mcp_server.tools_by_name[tool_name]


    if not tool_definition: # Should be caught by the check above if tools_by_name exists
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")

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
