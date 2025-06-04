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

    # Initialize source_iterable to an empty list by default.
    source_iterable = []

    # Attempt 1: Use mcp_server.list_tools() method if available (based on new log feedback)
    if hasattr(mcp_server, 'list_tools') and callable(getattr(mcp_server, 'list_tools')):
        print("Using async mcp_server.list_tools() method as primary source.")
        try:
            source_iterable = await mcp_server.list_tools() # <--- ADDED AWAIT
            # Ensure it's a list, in case it returns some other iterable type
            if not isinstance(source_iterable, list):
                print(f"mcp_server.list_tools() returned type {type(source_iterable)}, converting to list.")
                source_iterable = list(source_iterable)
        except Exception as e:
            print(f"Error calling await mcp_server.list_tools(): {e}")
            # Keep source_iterable empty so fallbacks are attempted
            source_iterable = [] # Ensure it's empty on error

    # Attempt 2: Fallback to mcp_server.tools (dict)
    if not source_iterable and hasattr(mcp_server, 'tools') and isinstance(mcp_server.tools, dict):
        print("Using mcp_server.tools.values() (from dict) as fallback.")
        source_iterable = list(mcp_server.tools.values()) # Ensure it's a list
    # Attempt 3: Fallback to mcp_server.tool_definitions (list)
    elif not source_iterable and hasattr(mcp_server, 'tool_definitions') and isinstance(mcp_server.tool_definitions, list):
        print("Using mcp_server.tool_definitions (list) as fallback.")
        source_iterable = mcp_server.tool_definitions
    # Attempt 4: Fallback to mcp_server.router.tools (list)
    elif not source_iterable and hasattr(mcp_server, 'router') and hasattr(mcp_server.router, 'tools') and isinstance(mcp_server.router.tools, list):
        print("Using mcp_server.router.tools (list) as fallback.")
        source_iterable = mcp_server.router.tools

    # If source_iterable is still empty after all attempts, then raise the error
    if not source_iterable:
        mcp_server_attrs = [attr for attr in dir(mcp_server) if not attr.startswith('_')]
        router_attrs = []
        if hasattr(mcp_server, 'router'):
            router_attrs = [attr for attr in dir(mcp_server.router) if not attr.startswith('_')]

        log_message = (
            "Critical: Could not find tool definitions on mcp_server after trying all known methods. "
            f"Checked for: mcp_server.list_tools() (callable), mcp_server.tools (dict), "
            f"mcp_server.tool_definitions (list), mcp_server.router.tools (list). "
            f"Available attributes on mcp_server: {mcp_server_attrs}. "
            f"Available attributes on mcp_server.router (if exists): {router_attrs}."
        )
        print(log_message) # This will go to server logs
        raise HTTPException(status_code=500, detail="Server configuration error: Cannot find tool definitions. Check server logs for details.")

    # The rest of the function continues here, processing source_iterable
    # tools_info_list = [] is initialized at the beginning of the function.
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
    tool_definition = None

    # Attempt 1: Use mcp_server.tool(tool_name) if available
    if hasattr(mcp_server, 'tool') and callable(getattr(mcp_server, 'tool')):
        print(f"Attempting to get tool '{tool_name}' schema using async mcp_server.tool().")
        try:
            tool_definition = await mcp_server.tool(tool_name) # <--- ADDED AWAIT
            if not tool_definition: # Explicitly check if mcp_server.tool() returned None
                 print(f"await mcp_server.tool('{tool_name}') returned None.")
        except Exception as e:
            print(f"Call to await mcp_server.tool('{tool_name}') failed: {e}")
            # tool_definition remains None, fallbacks will be attempted.

    # Attempt 2: Fallback to mcp_server.tools_by_name (dict)
    if not tool_definition and hasattr(mcp_server, 'tools_by_name') and isinstance(mcp_server.tools_by_name, dict):
        print(f"Attempting to get tool '{tool_name}' schema using mcp_server.tools_by_name.")
        tool_definition = mcp_server.tools_by_name.get(tool_name)

    # Attempt 3: Fallback to mcp_server.tools (dict)
    if not tool_definition and hasattr(mcp_server, 'tools') and isinstance(mcp_server.tools, dict):
        print(f"Attempting to get tool '{tool_name}' schema using mcp_server.tools.get().")
        tool_definition = mcp_server.tools.get(tool_name)

    # Attempt 4: Fallback to mcp_server.router.tools_by_name (dict)
    if not tool_definition and hasattr(mcp_server, 'router') and hasattr(mcp_server.router, 'tools_by_name') and isinstance(mcp_server.router.tools_by_name, dict):
        print(f"Attempting to get tool '{tool_name}' schema using mcp_server.router.tools_by_name.")
        tool_definition = mcp_server.router.tools_by_name.get(tool_name)

    # After all attempts, check if tool_definition was found
    if not tool_definition:
        mcp_server_attrs = [attr for attr in dir(mcp_server) if not attr.startswith('_')]
        log_message = (
            f"Critical: Tool '{tool_name}' for schema not found on mcp_server after trying all known methods. "
            f"Checked for: mcp_server.tool(), mcp_server.tools_by_name (dict), "
            f"mcp_server.tools (dict), mcp_server.router.tools_by_name (dict). "
            f"Available attributes on mcp_server: {mcp_server_attrs}."
        )
        print(log_message)
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found or server misconfigured for schema retrieval. Check server logs.")

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
    tool_name = request_data.tool_name # Ensure tool_name is defined from request_data
    args = request_data.args # Ensure args is defined

    tool_definition = None

    # Attempt 1: Use mcp_server.tool(tool_name) if available
    if hasattr(mcp_server, 'tool') and callable(getattr(mcp_server, 'tool')):
        print(f"Attempting to get tool '{tool_name}' for execution using async mcp_server.tool().")
        try:
            tool_definition = await mcp_server.tool(tool_name) # <--- ADDED AWAIT
            if not tool_definition: # Explicitly check if mcp_server.tool() returned None
                 print(f"await mcp_server.tool('{tool_name}') returned None.")
        except Exception as e:
            print(f"Call to await mcp_server.tool('{tool_name}') failed: {e}")
            # tool_definition remains None, fallbacks will be attempted.

    # Attempt 2: Fallback to mcp_server.tools_by_name (dict)
    if not tool_definition and hasattr(mcp_server, 'tools_by_name') and isinstance(mcp_server.tools_by_name, dict):
        print(f"Attempting to get tool '{tool_name}' for execution using mcp_server.tools_by_name.")
        tool_definition = mcp_server.tools_by_name.get(tool_name) # Use .get() for consistency

    # Attempt 3: Fallback to mcp_server.tools (dict)
    if not tool_definition and hasattr(mcp_server, 'tools') and isinstance(mcp_server.tools, dict):
        print(f"Attempting to get tool '{tool_name}' for execution using mcp_server.tools.get().")
        tool_definition = mcp_server.tools.get(tool_name)

    # Attempt 4: Fallback to mcp_server.router.tools_by_name (dict)
    if not tool_definition and hasattr(mcp_server, 'router') and hasattr(mcp_server.router, 'tools_by_name') and isinstance(mcp_server.router.tools_by_name, dict):
        print(f"Attempting to get tool '{tool_name}' for execution using mcp_server.router.tools_by_name.")
        tool_definition = mcp_server.router.tools_by_name.get(tool_name)

    # After all attempts, check if tool_definition was found
    if not tool_definition:
        mcp_server_attrs = [attr for attr in dir(mcp_server) if not attr.startswith('_')]
        log_message = (
            f"Critical: Tool '{tool_name}' for execution not found on mcp_server after trying all known methods. "
            f"Checked for: mcp_server.tool(), mcp_server.tools_by_name (dict), "
            f"mcp_server.tools (dict), mcp_server.router.tools_by_name (dict). "
            f"Available attributes on mcp_server: {mcp_server_attrs}."
        )
        print(log_message)
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found or server misconfigured for execution. Check server logs.")

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
