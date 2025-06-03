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
# Assuming 'mcp' (OpenRouterAI/model-context-protocol) is the one installed and working
from mcp.server.mcp import FastMCP # Changed from mcp.server.fastmcp
from mcp.server.context import Context # Changed from mcp.server.fastmcp
import numpy as np
from tensordirectory import storage # Assuming storage.py is in a package named tensordirectory
from pydantic import BaseModel, Field # Field might be useful for constraints/examples later
from typing import Optional, List, Dict, Any # Ensure List, Dict, Any are available

# Pydantic Models for Tool Arguments / Resource Payloads
class TensorUploadArgs(BaseModel):
    name: str
    description: str
    tensor_data: list # Using Python's built-in list type as per plan

class ModelUploadArgs(BaseModel):
    name: str
    description: str
    model_weights: Optional[list] = None # Using Python's built-in list type
    model_code: Optional[str] = None

# --- Pydantic Models for New Utility Tools ---

class DeleteArgs(BaseModel):
    name_or_uuid: str

class UpdateMetadataArgs(BaseModel):
    name_or_uuid: str
    # Allow arbitrary string-keyed dictionary for metadata updates.
    # Specific updatable fields will be handled in storage/tool logic.
    metadata_updates: Dict[str, Any]

class ListArgs(BaseModel):
    filter_by_name_contains: Optional[str] = None
    limit: int = Field(default=100, gt=0, le=1000) # Example constraints
    offset: int = Field(default=0, ge=0)

# --- Pydantic Models for Responses ---

class TensorMetadata(BaseModel):
    uuid: str
    user_name: str
    description: Optional[str] = None
    creation_date: Optional[str] = None # Kept as string from isoformat
    original_dtype: Optional[str] = None
    original_shape: Optional[str] = None

class ModelMetadata(BaseModel):
    uuid: str
    user_name: str
    description: Optional[str] = None
    upload_date: Optional[str] = None # Kept as string from isoformat
    has_code: bool
    has_weights: bool

class ListTensorsResponse(BaseModel):
    tensors: List[TensorMetadata]
    total_items_in_collection: int # Total items matching filter, not just in this page
    offset: int
    limit: int
    # count_in_response: int # Could add this: len(tensors)

class ListModelsResponse(BaseModel):
    models: List[ModelMetadata]
    total_items_in_collection: int # Total items matching filter
    offset: int
    limit: int
    # count_in_response: int

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


@mcp_server.tool()
async def upload_tensor(ctx: Context, args: TensorUploadArgs) -> dict:
    """
    MCP Tool for uploading a new tensor.

    Args:
        args: A TensorUploadArgs Pydantic model instance containing 'name',
              'description', and 'tensor_data'.
        ctx: The MCP Context object.

    Returns:
        A dictionary with operation status, UUID of the saved tensor, and name.
    """
    # Parameters are now accessed as attributes from the Pydantic model 'args'
    # Pydantic performs validation for presence and basic type of name, description, tensor_data.
    name = args.name
    description = args.description
    tensor_data_list = args.tensor_data

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

@mcp_server.tool()
async def upload_model(ctx: Context, args: ModelUploadArgs) -> dict:
    """
    MCP Tool for uploading a new model (code, weights, or both).

    Args:
        args: A ModelUploadArgs Pydantic model instance containing 'name',
              'description', and optionally 'model_weights' and/or 'model_code'.
        ctx: The MCP Context object.

    Returns:
        A dictionary with operation status, UUID of the saved model, and name.
    """
    # Required fields are validated by Pydantic.
    # Optional fields will be None if not provided.
    name = args.name
    description = args.description
    model_weights_list = args.model_weights
    model_code = args.model_code

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

# --- Helper for UUID check ---
import uuid as uuid_lib # Use alias to avoid conflict with storage.uuid
import h5py # Required for name resolution if not passing hf object around

def _is_uuid(value: str) -> bool:
    """Checks if a string value is a valid UUID."""
    try:
        uuid_lib.UUID(value)
        return True
    except ValueError:
        return False

# --- New Utility Tools ---

@mcp_server.tool()
async def list_tensors(ctx: Context, args: ListArgs) -> ListTensorsResponse:
    """Lists tensors with optional filtering by name and pagination."""
    await ctx.log_info(f"Listing tensors with filter: {args.filter_by_name_contains}, limit: {args.limit}, offset: {args.offset}")
    try:
        tensors_metadata_list, total_count = storage.list_tensors(
            filter_by_name_contains=args.filter_by_name_contains,
            limit=args.limit,
            offset=args.offset
        )

        # Convert list of dicts to list of TensorMetadata Pydantic models
        pydantic_tensors = [TensorMetadata(**meta) for meta in tensors_metadata_list]

        return ListTensorsResponse(
            tensors=pydantic_tensors,
            total_items_in_collection=total_count,
            offset=args.offset,
            limit=args.limit
        )
    except Exception as e:
        await ctx.log_error(f"Error listing tensors: {e}", exc_info=True)
        # Return an empty valid response on error, or could raise an MCP error
        return ListTensorsResponse(tensors=[], total_items_in_collection=0, offset=args.offset, limit=args.limit)

@mcp_server.tool()
async def list_models(ctx: Context, args: ListArgs) -> ListModelsResponse:
    """Lists models with optional filtering by name and pagination."""
    await ctx.log_info(f"Listing models with filter: {args.filter_by_name_contains}, limit: {args.limit}, offset: {args.offset}")
    try:
        models_metadata_list, total_count = storage.list_models(
            filter_by_name_contains=args.filter_by_name_contains,
            limit=args.limit,
            offset=args.offset
        )

        pydantic_models = [ModelMetadata(**meta) for meta in models_metadata_list]

        return ListModelsResponse(
            models=pydantic_models,
            total_items_in_collection=total_count,
            offset=args.offset,
            limit=args.limit
        )
    except Exception as e:
        await ctx.log_error(f"Error listing models: {e}", exc_info=True)
        return ListModelsResponse(models=[], total_items_in_collection=0, offset=args.offset, limit=args.limit)

@mcp_server.tool()
async def delete_tensor(ctx: Context, args: DeleteArgs) -> dict:
    """Deletes a tensor by its name or UUID."""
    await ctx.log_info(f"Attempting to delete tensor: {args.name_or_uuid}")
    item_to_delete = args.name_or_uuid
    actual_uuid = None

    if _is_uuid(item_to_delete):
        actual_uuid = item_to_delete
    else: # Assume it's a name, try to resolve it
        try:
            with h5py.File(storage.HDF5_FILE_NAME, 'r') as hf: # Read-only to get UUID
                resolved_uuid = storage._get_uuid_from_name(hf, "tensor", item_to_delete)
            if resolved_uuid:
                actual_uuid = resolved_uuid
            else:
                await ctx.log_info(f"Tensor with name '{item_to_delete}' not found for deletion.")
                return {"success": False, "message": f"Tensor '{item_to_delete}' not found by name."}
        except FileNotFoundError:
            await ctx.log_error(f"Storage file {storage.HDF5_FILE_NAME} not found during name resolution for delete_tensor.")
            return {"success": False, "message": "Storage file not found."}
        except Exception as e:
            await ctx.log_error(f"Error resolving tensor name '{item_to_delete}': {e}", exc_info=True)
            return {"success": False, "message": f"Error resolving tensor name: {str(e)}"}

    if not actual_uuid: # Should be caught by name resolution, but as a safeguard
         await ctx.log_error(f"Could not determine UUID for tensor '{item_to_delete}'.")
         return {"success": False, "message": f"Could not determine UUID for tensor '{item_to_delete}'."}

    try:
        success = storage.delete_tensor_by_uuid(actual_uuid)
        if success:
            await ctx.log_info(f"Tensor UUID '{actual_uuid}' (name/input: '{args.name_or_uuid}') deleted successfully.")
            return {"success": True, "message": f"Tensor '{args.name_or_uuid}' (UUID: {actual_uuid}) deleted successfully."}
        else:
            # This might mean the UUID was valid format but not found in storage.delete_tensor_by_uuid
            await ctx.log_info(f"Tensor UUID '{actual_uuid}' (name/input: '{args.name_or_uuid}') not found or delete failed in storage.")
            return {"success": False, "message": f"Tensor UUID '{actual_uuid}' not found or delete failed."}
    except Exception as e:
        await ctx.log_error(f"Exception deleting tensor UUID '{actual_uuid}': {e}", exc_info=True)
        return {"success": False, "message": f"An unexpected error occurred while deleting tensor: {str(e)}"}


@mcp_server.tool()
async def delete_model(ctx: Context, args: DeleteArgs) -> dict:
    """Deletes a model by its name or UUID."""
    await ctx.log_info(f"Attempting to delete model: {args.name_or_uuid}")
    item_to_delete = args.name_or_uuid
    actual_uuid = None

    if _is_uuid(item_to_delete):
        actual_uuid = item_to_delete
    else: # Assume it's a name
        try:
            with h5py.File(storage.HDF5_FILE_NAME, 'r') as hf:
                resolved_uuid = storage._get_uuid_from_name(hf, "model", item_to_delete)
            if resolved_uuid:
                actual_uuid = resolved_uuid
            else:
                await ctx.log_info(f"Model with name '{item_to_delete}' not found for deletion.")
                return {"success": False, "message": f"Model '{item_to_delete}' not found by name."}
        except FileNotFoundError:
            await ctx.log_error(f"Storage file {storage.HDF5_FILE_NAME} not found during name resolution for delete_model.")
            return {"success": False, "message": "Storage file not found."}
        except Exception as e:
            await ctx.log_error(f"Error resolving model name '{item_to_delete}': {e}", exc_info=True)
            return {"success": False, "message": f"Error resolving model name: {str(e)}"}

    if not actual_uuid:
         await ctx.log_error(f"Could not determine UUID for model '{item_to_delete}'.")
         return {"success": False, "message": f"Could not determine UUID for model '{item_to_delete}'."}

    try:
        success = storage.delete_model_by_uuid(actual_uuid)
        if success:
            await ctx.log_info(f"Model UUID '{actual_uuid}' (name/input: '{args.name_or_uuid}') deleted successfully.")
            return {"success": True, "message": f"Model '{args.name_or_uuid}' (UUID: {actual_uuid}) deleted successfully."}
        else:
            await ctx.log_info(f"Model UUID '{actual_uuid}' (name/input: '{args.name_or_uuid}') not found or delete failed in storage.")
            return {"success": False, "message": f"Model UUID '{actual_uuid}' not found or delete failed."}
    except Exception as e:
        await ctx.log_error(f"Exception deleting model UUID '{actual_uuid}': {e}", exc_info=True)
        return {"success": False, "message": f"An unexpected error occurred while deleting model: {str(e)}"}

@mcp_server.tool()
async def update_tensor_metadata(ctx: Context, args: UpdateMetadataArgs) -> dict:
    """Updates metadata for a tensor by its name or UUID."""
    await ctx.log_info(f"Attempting to update metadata for tensor: {args.name_or_uuid} with updates: {args.metadata_updates}")
    item_to_update = args.name_or_uuid
    actual_uuid = None

    if _is_uuid(item_to_update):
        actual_uuid = item_to_update
    else: # Assume it's a name
        try:
            with h5py.File(storage.HDF5_FILE_NAME, 'r') as hf:
                resolved_uuid = storage._get_uuid_from_name(hf, "tensor", item_to_update)
            if resolved_uuid:
                actual_uuid = resolved_uuid
            else:
                return {"success": False, "message": f"Tensor '{item_to_update}' not found by name for update."}
        except FileNotFoundError:
             await ctx.log_error(f"Storage file {storage.HDF5_FILE_NAME} not found during name resolution for update_tensor_metadata.")
             return {"success": False, "message": "Storage file not found."}
        except Exception as e:
            await ctx.log_error(f"Error resolving tensor name '{item_to_update}' for update: {e}", exc_info=True)
            return {"success": False, "message": f"Error resolving tensor name for update: {str(e)}"}

    if not actual_uuid:
         return {"success": False, "message": f"Could not determine UUID for tensor '{item_to_update}' for update."}

    try:
        updated_meta = storage.update_tensor_metadata(actual_uuid, args.metadata_updates)
        if updated_meta:
            await ctx.log_info(f"Metadata for tensor UUID '{actual_uuid}' updated successfully.")
            # Convert dict to TensorMetadata Pydantic model before returning if desired, or return dict
            # For consistency with list, let's return the Pydantic model if we have it
            # However, the storage function returns a dict. So, we'll stick to dict for now.
            # Or, we can convert: updated_meta_pydantic = TensorMetadata(**updated_meta)
            return {"success": True, "metadata": updated_meta}
        else:
            return {"success": False, "message": f"Tensor UUID '{actual_uuid}' not found or update failed."}
    except Exception as e:
        await ctx.log_error(f"Exception updating tensor metadata for UUID '{actual_uuid}': {e}", exc_info=True)
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}

@mcp_server.tool()
async def update_model_metadata(ctx: Context, args: UpdateMetadataArgs) -> dict:
    """Updates metadata for a model by its name or UUID."""
    await ctx.log_info(f"Attempting to update metadata for model: {args.name_or_uuid} with updates: {args.metadata_updates}")
    item_to_update = args.name_or_uuid
    actual_uuid = None

    if _is_uuid(item_to_update):
        actual_uuid = item_to_update
    else: # Assume it's a name
        try:
            with h5py.File(storage.HDF5_FILE_NAME, 'r') as hf:
                resolved_uuid = storage._get_uuid_from_name(hf, "model", item_to_update)
            if resolved_uuid:
                actual_uuid = resolved_uuid
            else:
                return {"success": False, "message": f"Model '{item_to_update}' not found by name for update."}
        except FileNotFoundError:
            await ctx.log_error(f"Storage file {storage.HDF5_FILE_NAME} not found during name resolution for update_model_metadata.")
            return {"success": False, "message": "Storage file not found."}
        except Exception as e:
            await ctx.log_error(f"Error resolving model name '{item_to_update}' for update: {e}", exc_info=True)
            return {"success": False, "message": f"Error resolving model name for update: {str(e)}"}

    if not actual_uuid:
        return {"success": False, "message": f"Could not determine UUID for model '{item_to_update}' for update."}

    try:
        updated_meta = storage.update_model_metadata(actual_uuid, args.metadata_updates)
        if updated_meta:
            await ctx.log_info(f"Metadata for model UUID '{actual_uuid}' updated successfully.")
            # updated_meta_pydantic = ModelMetadata(**updated_meta) # If converting to Pydantic model
            return {"success": True, "metadata": updated_meta}
        else:
            return {"success": False, "message": f"Model UUID '{actual_uuid}' not found or update failed."}
    except Exception as e:
        await ctx.log_error(f"Exception updating model metadata for UUID '{actual_uuid}': {e}", exc_info=True)
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}
