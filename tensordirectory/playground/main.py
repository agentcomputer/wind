from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any, List

# Import the MCP server instance and Context
from tensordirectory.mcp_interface import mcp_server
# from mcp.server.fastmcp import Context # Context might not be needed if Tool.execute_tool handles it
from pydantic import BaseModel, ValidationError # Import BaseModel and ValidationError

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
    tools_info = []
    for tool_name, tool_handler in mcp_server.tools.items():
        if hasattr(tool_handler, 'model') and tool_handler.model:
            schema = tool_handler.model.schema()
            parameters = []
            if 'properties' in schema:
                for param_name, param_info in schema.get('properties', {}).items():
                    parameters.append({
                        "name": param_name,
                        "type": param_info.get("type", "any"),
                        "title": param_info.get("title", param_name.replace("_", " ").title()),
                        "required": param_name in schema.get("required", [])
                    })
            tools_info.append({
                "name": tool_name,
                "description": schema.get("description", tool_handler.fn.__doc__ or "No description available."),
                "parameters": parameters
            })
        else:
            if tool_name == "query_tensor_directory":
                tools_info.append({
                    "name": tool_name,
                    "description": tool_handler.fn.__doc__ or "Query the tensor directory using natural language.",
                    "parameters": [
                        {"name": "prompt", "type": "string", "title": "Prompt", "required": True},
                        {"name": "params", "type": "string", "title": "Params (JSON string, optional)", "required": False}
                    ]
                })
            else:
                 tools_info.append({
                    "name": tool_name,
                    "description": tool_handler.fn.__doc__ or "No description available.",
                    "parameters": []
                })
    return JSONResponse(content={"tools": tools_info})

@app.get("/api/tools/{tool_name}/schema", response_class=JSONResponse)
async def get_tool_schema(tool_name: str):
    tool_handler = mcp_server.tools.get(tool_name)
    if not tool_handler:
        raise HTTPException(status_code=404, detail="Tool not found")

    if hasattr(tool_handler, 'model') and tool_handler.model:
        return JSONResponse(content=tool_handler.model.schema())
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
        return JSONResponse(content={"message": "This tool does not have a Pydantic arguments model."})

class ExecuteToolRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any]

@app.post("/api/execute_tool")
async def execute_tool_endpoint(request_data: ExecuteToolRequest):
    tool_name = request_data.tool_name
    args = request_data.args

    if tool_name not in mcp_server.tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")

    tool_definition = mcp_server.tools[tool_name]

    try:
        # The Tool.execute_tool method in FastMCP handles context creation and argument parsing.
        # It expects `args` to be a dictionary.
        # Pydantic validation errors during args parsing within execute_tool will be raised.
        result = await tool_definition.execute_tool(args)

        # The result from tool_definition.execute_tool might already be a Response object
        # or the direct data. If it's a FastMCP Response, we might want to extract its body.
        # For now, assume it's the direct data or a JSON-serializable dict.
        # If 'result' is a Pydantic model itself for some tools, FastAPI will handle serialization.
        return {"success": True, "result": result}

    except ValidationError as ve: # Catch Pydantic validation errors specifically
        # These errors occur if `args` are invalid for the tool's model
        # `tool_definition.execute_tool` should raise this if parsing fails.
        raise HTTPException(status_code=422, detail={"error_type": "Validation Error", "errors": ve.errors()})
    except HTTPException: # Re-raise HTTPExceptions if they were raised by the tool itself
        raise
    except Exception as e:
        # Catch any other exceptions during tool execution
        # Log the exception server-side for debugging
        print(f"Error executing tool '{tool_name}' with args {args}: {type(e).__name__} - {e}") # Basic logging
        # Return a generic error to the client
        raise HTTPException(status_code=500, detail={"error_type": str(type(e).__name__), "message": str(e)})

# Keep the uvicorn run command for direct execution if needed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tensordirectory.playground.main:app", host="0.0.0.0", port=8080, reload=True)
