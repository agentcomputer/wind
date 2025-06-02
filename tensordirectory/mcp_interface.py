"""
Defines the MCP (Model Context Protocol) interface for the TensorDirectory service.

This module sets up a FastMCP server and exposes resources for uploading tensors
and models, as well as a tool for querying the directory using natural language
processed by an AI agent. It relies on the `storage` module for data persistence
and the `agent` module for AI-driven query handling.
"""
# mcp_interface.py

# Ensure these dependencies are available in your environment:
# pip install modelcontextprotocol numpy h5py pydantic

from mcp.server.fastmcp import FastMCP, Context
import numpy as np
from tensordirectory import storage # Assuming storage.py is in a package named tensordirectory
from pydantic import BaseModel
from typing import Optional, List # List is used for more specific typing if needed, Optional for optional fields.

# Pydantic Models for Request Body Validation
class TensorUploadRequest(BaseModel):
    name: str
    description: str
    tensor_data: list # Using Python's built-in list type as per plan; can be List[any] essentially.
                      # For more specific structures like a 2D list of floats: List[List[float]]

class ModelUploadRequest(BaseModel):
    name: str
    description: str
    model_weights: Optional[list] = None # Using Python's built-in list type.
    model_code: Optional[str] = None

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
async def upload_tensor_resource(data: TensorUploadRequest, ctx: Context) -> dict:
    """
    MCP Resource handler for uploading a new tensor.

    Args:
        data: A TensorUploadRequest Pydantic model instance containing 'name', 
              'description', and 'tensor_data'.
        ctx: The MCP Context object.

    Returns:
        A dictionary with operation status, UUID of the saved tensor, and name.
    """
    # Parameters are now accessed as attributes from the Pydantic model 'data'
    # Pydantic performs validation for presence and basic type of name, description, tensor_data.
    name = data.name
    description = data.description
    tensor_data_list = data.tensor_data 

    await ctx.log_info(f"Attempting to upload tensor: {name}")
    try:
        # Pydantic ensures tensor_data_list is a list.
        # Still, we might want to check if it's an empty list if that's not allowed.
        if not tensor_data_list: 
            await ctx.log_error(f"tensor_data for '{name}' must be a non-empty list.")
            return {"error": f"tensor_data for '{name}' must be a non-empty list."}
        
        arr = np.array(tensor_data_list)
        # Further validation can be added here (e.g., check dimensions, specific dtype compatibility)
        
        tensor_uuid = storage.save_tensor(name=name, description=description, tensor_data=arr)
        if tensor_uuid:
            await ctx.log_info(f"Tensor '{name}' saved with UUID: {tensor_uuid}")
            return {"uuid": tensor_uuid, "name": name, "message": "Tensor uploaded successfully"}
        else:
            await ctx.log_error(f"Failed to save tensor '{name}' - storage returned no UUID")
            return {"error": f"Failed to save tensor '{name}'"}
    except ValueError as ve: # Handles errors from np.array conversion if data is malformed for numpy
        await ctx.log_error(f"ValueError during tensor conversion for '{name}': {ve}", exc_info=True)
        return {"error": f"Invalid tensor data format for '{name}': {ve}"}
    except Exception as e: # Catch-all for other unexpected errors during storage or processing
        await ctx.log_error(f"Exception uploading tensor '{name}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while uploading tensor '{name}': {str(e)}"}

@mcp_server.resource("tensordirectory://models/upload")
async def upload_model_resource(data: ModelUploadRequest, ctx: Context) -> dict:
    """
    MCP Resource handler for uploading a new model (code, weights, or both).

    Args:
        data: A ModelUploadRequest Pydantic model instance containing 'name', 
              'description', and optionally 'model_weights' and/or 'model_code'.
        ctx: The MCP Context object.

    Returns:
        A dictionary with operation status, UUID of the saved model, and name.
    """
    # Required fields are validated by Pydantic.
    # Optional fields will be None if not provided.
    name = data.name
    description = data.description
    model_weights_list = data.model_weights 
    model_code = data.model_code

    await ctx.log_info(f"Attempting to upload model: {name}")
    try:
        if model_weights_list is None and model_code is None:
            # This check remains important as both are optional in the Pydantic model,
            # but the application requires at least one.
            await ctx.log_error(f"Validation error for model '{name}': Either model_weights or model_code must be provided.")
            return {"error": "Either model_weights or model_code must be provided."}

        np_weights = None
        if model_weights_list is not None: # Check if model_weights_list was provided
            # Pydantic ensures model_weights_list is a list if provided.
            # An empty list for weights might be valid (e.g. representing no weights explicitly)
            # or could be an error depending on deeper application logic.
            # For now, if it's an empty list, np.array([]) is valid.
            np_weights = np.array(model_weights_list)

        model_uuid = storage.save_model(name=name, description=description, model_weights=np_weights, model_code=model_code)
        if model_uuid:
            await ctx.log_info(f"Model '{name}' saved with UUID: {model_uuid}")
            return {"uuid": model_uuid, "name": name, "message": "Model uploaded successfully"}
        else:
            await ctx.log_error(f"Failed to save model '{name}' - storage returned no UUID")
            return {"error": f"Failed to save model '{name}'"}
    except ValueError as ve: # Handles errors from np.array conversion if data is malformed
        await ctx.log_error(f"ValueError during model weights conversion for '{name}': {ve}", exc_info=True)
        return {"error": f"Invalid model_weights data format for '{name}': {ve}"}
    except Exception as e: # Catch-all for other unexpected errors
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
