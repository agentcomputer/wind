from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any, List

# Import the MCP server instance
from tensordirectory.mcp_interface import mcp_server # mcp_server is an instance of mcp.server.fastmcp.FastMCP
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
    Uses mcp_server.tool_definitions for iteration.
    """
    tools_info_list = []

    if not hasattr(mcp_server, 'tool_definitions'):
        raise HTTPException(status_code=500, detail="Server configuration error: 'tool_definitions' not found on mcp_server.")

    for tool_def in mcp_server.tool_definitions:
        tool_name = tool_def.name
        description = tool_def.description or tool_def.fn.__doc__ or "No description available."

        parameters = []
        pydantic_model = tool_def.model

        if pydantic_model:
            schema = pydantic_model.schema()
            if 'properties' in schema:
                for param_name, param_info in schema.get('properties', {}).items():
                    parameters.append({
                        "name": param_name,
                        "type": param_info.get("type", "any"),
                        "title": param_info.get("title", param_name.replace("_", " ").title()),
                        "required": param_name in schema.get("required", [])
                    })
        elif tool_name == "query_tensor_directory":
             description = tool_def.fn.__doc__ or "Query the tensor directory using natural language."
             parameters = [
                {"name": "prompt", "type": "string", "title": "Prompt", "required": True},
                # For query_tensor_directory, 'params' is a direct kwarg, not part of a Pydantic model
                # Representing it as an object that can take arbitrary key-value pairs
                {"name": "params", "type": "object", "title": "Params (JSON object, optional)", "required": False, "default": None}
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
    Uses mcp_server.tools.get(tool_name) for lookup.
    """
    if not hasattr(mcp_server, 'tools') or not isinstance(mcp_server.tools, dict):
        raise HTTPException(status_code=500, detail="Server configuration error: 'tools' dictionary not found on mcp_server.")

    tool_instance = mcp_server.tools.get(tool_name)

    if not tool_instance:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")

    if tool_instance.model:
        return JSONResponse(content=tool_instance.model.schema())
    elif tool_name == "query_tensor_directory":
        # Manual schema for query_tensor_directory if its model is None
        # but it accepts 'prompt' and 'params' directly.
        return JSONResponse(content={
            "title": "QueryTensorDirectoryArgs",
            "type": "object",
            "properties": {
                "prompt": {"title": "Prompt", "type": "string"},
                "params": {"title": "Params", "type": "object", "default": None, "description": "JSON object for additional parameters. Optional."}
            },
            "required": ["prompt"]
        })
    else:
        return JSONResponse(content={"message": f"Tool '{tool_name}' does not have a defined Pydantic arguments model for schema generation."})

class ExecuteToolRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any]

@app.post("/api/execute_tool")
async def execute_tool_endpoint(request_data: ExecuteToolRequest):
    tool_name = request_data.tool_name
    args = request_data.args

    if not hasattr(mcp_server, 'tools') or not isinstance(mcp_server.tools, dict):
        raise HTTPException(status_code=500, detail="Server configuration error: 'tools' dictionary not found for execution.")

    if tool_name not in mcp_server.tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")

    tool_instance = mcp_server.tools[tool_name]

    try:
        result = await tool_instance.execute_tool(args)
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
