"""
Defines the MCP (Model Context Protocol) interface for the TensorDirectory service.

This module sets up a FastMCP server and exposes resources for uploading tensors
and models, as well as a tool for querying the directory using natural language
processed by an AI agent. It relies on the `storage` module for data persistence
and the `agent` module for AI-driven query handling.
"""
# mcp_interface.py

# Ensure these dependencies are available in your environment:
# pip install mcp numpy h5py pydantic

# Corrected imports for OpenRouterAI mcp package (mcp-1.9.2)
from mcp.server import FastMCP # FastMCP is exposed by mcp.server's __init__.py
from mcp.context import Context   # Context is in mcp/context.py (top-level mcp package)
import numpy as np
from tensordirectory import storage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Pydantic Models for Tool Arguments / Resource Payloads
class TensorUploadArgs(BaseModel):
    name: str
    description: str
    tensor_data: list

class ModelUploadArgs(BaseModel):
    name: str
    description: str
    model_weights: Optional[list] = None
    model_code: Optional[str] = None

class DeleteArgs(BaseModel):
    name_or_uuid: str

class UpdateMetadataArgs(BaseModel):
    name_or_uuid: str
    metadata_updates: Dict[str, Any]

class ListArgs(BaseModel):
    filter_by_name_contains: Optional[str] = None
    limit: int = Field(default=100, gt=0, le=1000)
    offset: int = Field(default=0, ge=0)

class TensorMetadata(BaseModel):
    uuid: str
    user_name: str
    description: Optional[str] = None
    creation_date: Optional[str] = None
    original_dtype: Optional[str] = None
    original_shape: Optional[str] = None

class ModelMetadata(BaseModel):
    uuid: str
    user_name: str
    description: Optional[str] = None
    upload_date: Optional[str] = None
    has_code: bool
    has_weights: bool

class ListTensorsResponse(BaseModel):
    tensors: List[TensorMetadata]
    total_items_in_collection: int
    offset: int
    limit: int

class ListModelsResponse(BaseModel):
    models: List[ModelMetadata]
    total_items_in_collection: int
    offset: int
    limit: int

mcp_server = FastMCP(
    "TensorDirectoryService",
    dependencies=["numpy", "h5py"]
)

async def invoke_agent_query(prompt: str, params: dict | None, ctx: Context) -> str:
    await ctx.log_info(f"Agent query received for prompt: {prompt}. Params: {params}")
    from . import agent
    return await agent.handle_query(user_prompt=prompt, params=params, ctx=ctx)

@mcp_server.tool()
async def upload_tensor(ctx: Context, args: TensorUploadArgs) -> dict:
    name = args.name
    description = args.description
    tensor_data_list = args.tensor_data
    await ctx.log_info(f"Attempting to upload tensor: {name}")
    try:
        if not tensor_data_list:
            await ctx.log_error(f"tensor_data for '{name}' must be a non-empty list.")
            return {"error": f"tensor_data for '{name}' must be a non-empty list."}
        arr = np.array(tensor_data_list)
        tensor_uuid = storage.save_tensor(name=name, description=description, tensor_data=arr)
        if tensor_uuid:
            await ctx.log_info(f"Tensor '{name}' saved with UUID: {tensor_uuid}")
            return {"uuid": tensor_uuid, "name": name, "message": "Tensor uploaded successfully"}
        else:
            await ctx.log_error(f"Failed to save tensor '{name}' - storage returned no UUID")
            return {"error": f"Failed to save tensor '{name}'"}
    except ValueError as ve:
        await ctx.log_error(f"ValueError during tensor conversion for '{name}': {ve}", exc_info=True)
        return {"error": f"Invalid tensor data format for '{name}': {ve}"}
    except Exception as e:
        await ctx.log_error(f"Exception uploading tensor '{name}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while uploading tensor '{name}': {str(e)}"}

@mcp_server.tool()
async def upload_model(ctx: Context, args: ModelUploadArgs) -> dict:
    name = args.name
    description = args.description
    model_weights_list = args.model_weights
    model_code = args.model_code
    await ctx.log_info(f"Attempting to upload model: {name}")
    try:
        if model_weights_list is None and model_code is None:
            await ctx.log_error(f"Validation error for model '{name}': Either model_weights or model_code must be provided.")
            return {"error": "Either model_weights or model_code must be provided."}
        np_weights = None
        if model_weights_list is not None:
            np_weights = np.array(model_weights_list)
        model_uuid = storage.save_model(name=name, description=description, model_weights=np_weights, model_code=model_code)
        if model_uuid:
            await ctx.log_info(f"Model '{name}' saved with UUID: {model_uuid}")
            return {"uuid": model_uuid, "name": name, "message": "Model uploaded successfully"}
        else:
            await ctx.log_error(f"Failed to save model '{name}' - storage returned no UUID")
            return {"error": f"Failed to save model '{name}'"}
    except ValueError as ve:
        await ctx.log_error(f"ValueError during model weights conversion for '{name}': {ve}", exc_info=True)
        return {"error": f"Invalid model_weights data format for '{name}': {ve}"}
    except Exception as e:
        await ctx.log_error(f"Exception uploading model '{name}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while uploading model '{name}': {str(e)}"}

@mcp_server.tool()
async def query_tensor_directory(prompt: str, ctx: Context, params: dict | None = None) -> str:
    await ctx.log_info(f"Query received for TensorDirectory: '{prompt}' with params: {params}")
    response = await invoke_agent_query(prompt=prompt, params=params, ctx=ctx)
    return response

import uuid as uuid_lib
import h5py

def _is_uuid(value: str) -> bool:
    try:
        uuid_lib.UUID(value)
        return True
    except ValueError:
        return False

@mcp_server.tool()
async def list_tensors(ctx: Context, args: ListArgs) -> ListTensorsResponse:
    await ctx.log_info(f"Listing tensors with filter: {args.filter_by_name_contains}, limit: {args.limit}, offset: {args.offset}")
    try:
        tensors_metadata_list, total_count = storage.list_tensors(
            filter_by_name_contains=args.filter_by_name_contains,
            limit=args.limit,
            offset=args.offset
        )
        pydantic_tensors = [TensorMetadata(**meta) for meta in tensors_metadata_list]
        return ListTensorsResponse(
            tensors=pydantic_tensors,
            total_items_in_collection=total_count,
            offset=args.offset,
            limit=args.limit
        )
    except Exception as e:
        await ctx.log_error(f"Error listing tensors: {e}", exc_info=True)
        return ListTensorsResponse(tensors=[], total_items_in_collection=0, offset=args.offset, limit=args.limit)

@mcp_server.tool()
async def list_models(ctx: Context, args: ListArgs) -> ListModelsResponse:
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
    await ctx.log_info(f"Attempting to delete tensor: {args.name_or_uuid}")
    item_to_delete = args.name_or_uuid
    actual_uuid = None
    if _is_uuid(item_to_delete):
        actual_uuid = item_to_delete
    else:
        try:
            with h5py.File(storage.HDF5_FILE_NAME, 'r') as hf:
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
    if not actual_uuid:
         await ctx.log_error(f"Could not determine UUID for tensor '{item_to_delete}'.")
         return {"success": False, "message": f"Could not determine UUID for tensor '{item_to_delete}'."}
    try:
        success = storage.delete_tensor_by_uuid(actual_uuid)
        if success:
            await ctx.log_info(f"Tensor UUID '{actual_uuid}' (name/input: '{args.name_or_uuid}') deleted successfully.")
            return {"success": True, "message": f"Tensor '{args.name_or_uuid}' (UUID: {actual_uuid}) deleted successfully."}
        else:
            await ctx.log_info(f"Tensor UUID '{actual_uuid}' (name/input: '{args.name_or_uuid}') not found or delete failed in storage.")
            return {"success": False, "message": f"Tensor UUID '{actual_uuid}' not found or delete failed."}
    except Exception as e:
        await ctx.log_error(f"Exception deleting tensor UUID '{actual_uuid}': {e}", exc_info=True)
        return {"success": False, "message": f"An unexpected error occurred while deleting tensor: {str(e)}"}

@mcp_server.tool()
async def delete_model(ctx: Context, args: DeleteArgs) -> dict:
    await ctx.log_info(f"Attempting to delete model: {args.name_or_uuid}")
    item_to_delete = args.name_or_uuid
    actual_uuid = None
    if _is_uuid(item_to_delete):
        actual_uuid = item_to_delete
    else:
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
    await ctx.log_info(f"Attempting to update metadata for tensor: {args.name_or_uuid} with updates: {args.metadata_updates}")
    item_to_update = args.name_or_uuid
    actual_uuid = None
    if _is_uuid(item_to_update):
        actual_uuid = item_to_update
    else:
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
            return {"success": True, "metadata": updated_meta}
        else:
            return {"success": False, "message": f"Tensor UUID '{actual_uuid}' not found or update failed."}
    except Exception as e:
        await ctx.log_error(f"Exception updating tensor metadata for UUID '{actual_uuid}': {e}", exc_info=True)
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}

@mcp_server.tool()
async def update_model_metadata(ctx: Context, args: UpdateMetadataArgs) -> dict:
    await ctx.log_info(f"Attempting to update metadata for model: {args.name_or_uuid} with updates: {args.metadata_updates}")
    item_to_update = args.name_or_uuid
    actual_uuid = None
    if _is_uuid(item_to_update):
        actual_uuid = item_to_update
    else:
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
            return {"success": True, "metadata": updated_meta}
        else:
            return {"success": False, "message": f"Model UUID '{actual_uuid}' not found or update failed."}
    except Exception as e:
        await ctx.log_error(f"Exception updating model metadata for UUID '{actual_uuid}': {e}", exc_info=True)
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}
