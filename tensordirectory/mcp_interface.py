"""
Defines the MCP (Model Context Protocol) interface for the TensorDirectory service.

This module sets up a FastMCP server and exposes resources for uploading tensors
and models, as well as a tool for querying the directory using natural language
processed by an AI agent. It relies on the `storage` module for data persistence
and the `agent` module for AI-driven query handling.
"""
# mcp_interface.py

# Ensure these dependencies are available in your environment:
# pip install modelcontextprotocol numpy h5py

from mcp.server.fastmcp import FastMCP, Context
import numpy as np
from tensordirectory import storage # Assuming storage.py is in a package named tensordirectory

# Initialize MCP Server
mcp_server = FastMCP(
    "TensorDirectoryService",
    dependencies=["numpy", "h5py"] # Listing dependencies for the server
)

# Placeholder for agent interaction, to be implemented in agent.py
async def invoke_agent_query(prompt: str, params: dict | None, ctx: Context) -> str:
    """
    Placeholder function to simulate invoking the AI agent.
    In a complete implementation, this would call `agent.handle_query`.
    """
    # This will eventually call the Gemini model and orchestrate the response.
    # For now, it's a placeholder.
    # from . import agent # Lazy import or pass agent instance if needed
    # return await agent.handle_query(prompt, params, ctx) # This is the target call
    await ctx.log_info(f"Agent query received for prompt: {prompt}. Params: {params}")
    # The actual agent.py's handle_query is now implemented, so this could be updated
    # to call it if the agent module was fully integrated here.
    # For now, keeping as a distinct placeholder as per original structure for mcp_interface.
    # If agent.py is available in sys.path:
    from . import agent 
    return await agent.handle_query(user_prompt=prompt, params=params, ctx=ctx)


@mcp_server.resource("tensordirectory://tensors/upload")
async def upload_tensor_resource(data: dict, ctx: Context) -> dict:
    """
    MCP Resource handler for uploading a new tensor.

    Args:
        data: A dictionary containing 'name', 'description', and 'tensor_data'.
        ctx: The MCP Context object.

    Returns:
        A dictionary with operation status, UUID of the saved tensor, and name.
    """
    try:
        name = data['name']
        description = data['description']
        tensor_data_list = data['tensor_data'] # Keep original name for clarity before conversion
    except KeyError as e:
        await ctx.log_error(f"Missing key in request data for upload_tensor_resource: {e}", exc_info=True)
        return {"error": f"Missing required field: {str(e)}"}

    await ctx.log_info(f"Attempting to upload tensor: {name}")
    try:
        # Basic validation for tensor_data
        if not isinstance(tensor_data_list, list) or not tensor_data_list:
            return {"error": "tensor_data must be a non-empty list"}
        
        arr = np.array(tensor_data_list)
        # Further validation can be added here (e.g., check dimensions, type)
        
        tensor_uuid = storage.save_tensor(name=name, description=description, tensor_data=arr) # Changed variable name from uuid to tensor_uuid
        if tensor_uuid:
            await ctx.log_info(f"Tensor '{name}' saved with UUID: {tensor_uuid}")
            return {"uuid": tensor_uuid, "name": name, "message": "Tensor uploaded successfully"}
        else:
            await ctx.log_error(f"Failed to save tensor '{name}' - storage returned no UUID")
            return {"error": f"Failed to save tensor '{name}'"}
    except ValueError as ve:
        await ctx.log_error(f"ValueError during tensor conversion for '{name}': {ve}", exc_info=True) # Added exc_info
        return {"error": f"Invalid tensor data format for '{name}': {ve}"}
    except Exception as e:
        await ctx.log_error(f"Exception uploading tensor '{name}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while uploading tensor '{name}': {str(e)}"}

@mcp_server.resource("tensordirectory://models/upload")
async def upload_model_resource(data: dict, ctx: Context) -> dict:
    """
    MCP Resource handler for uploading a new model (code, weights, or both).

    Args:
        data: A dictionary containing 'name', 'description', and optionally
              'model_weights' and/or 'model_code'.
        ctx: The MCP Context object.

    Returns:
        A dictionary with operation status, UUID of the saved model, and name.
    """
    try:
        name = data['name']
        description = data['description']
    except KeyError as e:
        await ctx.log_error(f"Missing required key in request data for upload_model_resource: {e}", exc_info=True)
        return {"error": f"Missing required field: {str(e)}"}

    # Optional fields
    model_weights_list = data.get('model_weights') # Keep original name for clarity before conversion
    model_code = data.get('model_code')

    await ctx.log_info(f"Attempting to upload model: {name}")
    try:
        if model_weights_list is None and model_code is None:
            return {"error": "Either model_weights or model_code must be provided."}

        np_weights = None
        if model_weights_list:
            if not isinstance(model_weights_list, list): # Basic validation
                 return {"error": "model_weights must be a list if provided"}
            np_weights = np.array(model_weights_list)

        model_uuid = storage.save_model(name=name, description=description, model_weights=np_weights, model_code=model_code)
        if model_uuid: # Changed variable name from uuid to model_uuid
            await ctx.log_info(f"Model '{name}' saved with UUID: {model_uuid}")
            return {"uuid": model_uuid, "name": name, "message": "Model uploaded successfully"}
        else:
            await ctx.log_error(f"Failed to save model '{name}' - storage returned no UUID")
            return {"error": f"Failed to save model '{name}'"}
    except ValueError as ve:
        await ctx.log_error(f"ValueError during model weights conversion for '{name}': {ve}", exc_info=True) # Added exc_info
        return {"error": f"Invalid model_weights data format for '{name}': {ve}"}
    except Exception as e:
        await ctx.log_error(f"Exception uploading model '{name}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while uploading model '{name}': {str(e)}"}

@mcp_server.tool()
async def query_tensor_directory(prompt: str, ctx: Context, params: dict | None = None) -> str:
    """
    Processes a user's prompt to query the tensor directory.
    The AI agent will interpret the prompt, retrieve data, or use an inference model.
    """
    await ctx.log_info(f"Query received for TensorDirectory: '{prompt}' with params: {params}")
    # This will call the agent logic implemented in agent.py
    # The invoke_agent_query function has been updated to call the actual agent.handle_query
    response = await invoke_agent_query(prompt=prompt, params=params, ctx=ctx)
    return response

# Example of how to potentially run this server (though main.py will handle actual execution)
# if __name__ == "__main__":
#     # This is for demonstration; actual server running will be in main.py
#     # You would need to ensure tensordirectory.storage can be imported.
#     # e.g., by running from the parent directory of 'tensordirectory'
#     # or by installing the package.
#     print("MCP Server defined. Run main.py to start.")
#     # For testing this file directly (if paths are set up):
#     # mcp_server.run()
