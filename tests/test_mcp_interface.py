# tests/test_mcp_interface.py
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
# AsyncMock for async functions, MagicMock for synchronous or context
import numpy as np
import asyncio # Required for IsolatedAsyncioTestCase if not using it directly

# Assuming tensordirectory is in PYTHONPATH or project is structured as a package
from tensordirectory.mcp_interface import (
    upload_tensor,
    upload_model,
    query_tensor_directory,
    # New tools to import for testing
    list_tensors,
    list_models,
    delete_tensor,
    delete_model,
    update_tensor_metadata,
    update_model_metadata,
    mcp_server,
    TensorUploadArgs,
    ModelUploadArgs,
    # New Pydantic models for args and responses to import
    ListArgs,
    DeleteArgs,
    UpdateMetadataArgs,
    TensorMetadata,
    ModelMetadata,
    ListTensorsResponse,
    ListModelsResponse
)
from mcp.server.fastmcp import FastMCP, Context # For type checking and mock context
import tensordirectory.mcp_interface # To patch h5py.File within this module's scope

# It's often easier to mock the specific functions from storage that mcp_interface uses
# rather than the whole module.

class TestMCPInterface(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Common mock context for all tests in this class
        self.mock_ctx = MagicMock(spec=Context)
        self.mock_ctx.log_info = AsyncMock()
        self.mock_ctx.log_error = AsyncMock()
        self.mock_ctx.log_warning = AsyncMock() # If any handler uses log_warning

    async def test_01_mcp_server_instance(self):
        self.assertIsInstance(mcp_server, FastMCP)

    @patch('tensordirectory.mcp_interface.storage.save_tensor', new_callable=MagicMock)
    async def test_02_upload_tensor_resource_success(self, mock_save_tensor):
        mock_uuid = "test-tensor-uuid-123"
        mock_save_tensor.return_value = mock_uuid

        tensor_list_data = [[1, 2], [3, 4]]
        request_args = TensorUploadArgs( # Use updated model name
            name="test_tensor",
            description="A test tensor",
            tensor_data=tensor_list_data
        )
        response = await upload_tensor(self.mock_ctx, args=request_args) # Call updated function name and use args=

        mock_save_tensor.assert_called_once()
        args, kwargs = mock_save_tensor.call_args_list[0] # Get the first call's arguments
        # In this case, args is empty, kwargs contains the named arguments
        self.assertIsInstance(kwargs['tensor_data'], np.ndarray)
        np.testing.assert_array_equal(kwargs['tensor_data'], np.array(tensor_list_data))
        self.assertEqual(kwargs['name'], "test_tensor")
        self.assertEqual(kwargs['description'], "A test tensor")

        self.assertEqual(response, {"uuid": mock_uuid, "name": "test_tensor", "message": "Tensor uploaded successfully"})
        self.mock_ctx.log_info.assert_any_call("Attempting to upload tensor: test_tensor")
        self.mock_ctx.log_info.assert_any_call(f"Tensor 'test_tensor' saved with UUID: {mock_uuid}")


    @patch('tensordirectory.mcp_interface.storage.save_tensor', new_callable=MagicMock)
    async def test_03_upload_tensor_resource_invalid_data_empty(self, mock_save_tensor):
        request_args = TensorUploadArgs(name="invalid_tensor", description="Invalid", tensor_data=[]) # Use updated model name
        response = await upload_tensor(self.mock_ctx, args=request_args) # Call updated function name and use args=

        self.assertIn("error", response)
        # The error message comes from within the handler after Pydantic validation passes for `list` type
        self.assertEqual(response["error"], "tensor_data for 'invalid_tensor' must be a non-empty list.")
        mock_save_tensor.assert_not_called()
        self.mock_ctx.log_info.assert_any_call("Attempting to upload tensor: invalid_tensor")
        self.mock_ctx.log_error.assert_called_with("tensor_data for 'invalid_tensor' must be a non-empty list.")

    @patch('tensordirectory.mcp_interface.storage.save_tensor') # Outermost mock, new_callable defaults to MagicMock
    @patch('tensordirectory.mcp_interface.np.array')      # Innermost mock
    async def test_03b_upload_tensor_resource_value_error_on_conversion(self, mock_np_array, mock_save_tensor): # Order matters
        mock_np_array.side_effect = ValueError("Simulated np.array error")

        request_args = TensorUploadArgs(
            name="value_error_tensor",
            description="Numpy conversion failure test",
            # Data can be anything, as np.array is mocked to fail
            tensor_data=[[1,2]]
        )

        response = await upload_tensor(self.mock_ctx, args=request_args)

        self.assertIn("error", response)
        # Check if the simulated error message is part of the response
        self.assertIn("Invalid tensor data format for 'value_error_tensor': Simulated np.array error", response["error"])
        mock_save_tensor.assert_not_called()
        self.mock_ctx.log_error.assert_any_call(
            "ValueError during tensor conversion for 'value_error_tensor': Simulated np.array error",
            exc_info=True
        )

    # test_03c_upload_tensor_resource_missing_keys is removed as Pydantic handles this.
    # FastMCP would typically return a 422 error if Pydantic validation fails.

    @patch('tensordirectory.mcp_interface.storage.save_tensor', return_value=None)
    async def test_04_upload_tensor_resource_storage_failure(self, mock_save_tensor_failure):
        request_args = TensorUploadArgs(name="fail_tensor", description="Fail save", tensor_data=[[1]]) # Use updated model name
        response = await upload_tensor(self.mock_ctx, args=request_args) # Call updated function name and use args=

        self.assertIn("error", response)
        self.assertEqual(response["error"], "Failed to save tensor 'fail_tensor'")
        mock_save_tensor_failure.assert_called_once()
        self.mock_ctx.log_error.assert_called_with("Failed to save tensor 'fail_tensor' - storage returned no UUID")


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_05_upload_model_resource_success_code_only(self, mock_save_model):
        mock_uuid = "test-model-uuid-456"
        mock_save_model.return_value = mock_uuid

        request_args = ModelUploadArgs( # Use updated model name
            name="code_model",
            description="Code only",
            model_code="print('hi')"
        )
        response = await upload_model(self.mock_ctx, args=request_args) # Call updated function name and use args=

        mock_save_model.assert_called_once_with(name="code_model", description="Code only", model_weights=None, model_code="print('hi')")
        self.assertEqual(response, {"uuid": mock_uuid, "name": "code_model", "message": "Model uploaded successfully"})
        self.mock_ctx.log_info.assert_any_call("Attempting to upload model: code_model")
        self.mock_ctx.log_info.assert_any_call(f"Model 'code_model' saved with UUID: {mock_uuid}")


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_06_upload_model_resource_success_weights_only(self, mock_save_model):
        mock_uuid = "test-model-uuid-789"
        mock_save_model.return_value = mock_uuid

        weights_list = [[1.0, 2.0], [3.0, 4.0]]
        request_args = ModelUploadArgs( # Use updated model name
            name="weights_model",
            description="Weights only",
            model_weights=weights_list
        )
        response = await upload_model(self.mock_ctx, args=request_args) # Call updated function name and use args=

        args, kwargs = mock_save_model.call_args_list[0]
        self.assertIsInstance(kwargs['model_weights'], np.ndarray)
        np.testing.assert_array_equal(kwargs['model_weights'], np.array(weights_list))
        self.assertEqual(kwargs['name'], "weights_model")
        self.assertIsNone(kwargs['model_code'])
        self.assertEqual(response, {"uuid": mock_uuid, "name": "weights_model", "message": "Model uploaded successfully"})

    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_06b_upload_model_resource_success_code_and_weights(self, mock_save_model):
        mock_uuid = "test-model-uuid-cw-123"
        mock_save_model.return_value = mock_uuid
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock()
        mock_ctx.log_error = AsyncMock()

        weights_list = [[1.0, 2.0]]
        code_str = "def predict(): pass"
        weights_list_for_cw = [[1.0, 2.0]] # Use a different var name to avoid conflict if setUp uses it
        request_args = ModelUploadArgs( # Use updated model name
            name="cw_model",
            description="Code & Weights",
            model_weights=weights_list_for_cw,
            model_code=code_str
        )
        response = await upload_model(self.mock_ctx, args=request_args) # Call updated function name and use args=

        args, kwargs = mock_save_model.call_args_list[0]
        self.assertIsInstance(kwargs['model_weights'], np.ndarray)
        np.testing.assert_array_equal(kwargs['model_weights'], np.array(weights_list))
        self.assertEqual(kwargs['model_code'], code_str)
        self.assertEqual(kwargs['name'], "cw_model")
        self.assertEqual(response, {"uuid": mock_uuid, "name": "cw_model", "message": "Model uploaded successfully"})


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_07_upload_model_resource_no_code_no_weights(self, mock_save_model):
        # self.mock_ctx is used from setUp
        request_args = ModelUploadArgs(name="empty_model", description="Empty") # No model_code or model_weights
        response = await upload_model(self.mock_ctx, args=request_args) # Call updated function name and use args=

        self.assertIn("error", response)
        self.assertEqual(response["error"], "Either model_weights or model_code must be provided.")
        mock_save_model.assert_not_called()
        self.mock_ctx.log_info.assert_any_call("Attempting to upload model: empty_model")
        self.mock_ctx.log_error.assert_called_with("Validation error for model 'empty_model': Either model_weights or model_code must be provided.")


    @patch('tensordirectory.mcp_interface.storage.save_model') # Outermost mock
    @patch('tensordirectory.mcp_interface.np.array')      # Innermost mock
    async def test_07b_upload_model_resource_value_error_on_conversion(self, mock_np_array, mock_save_model): # Order matters
        mock_np_array.side_effect = ValueError("Simulated np.array error for model weights")

        request_args = ModelUploadArgs(
            name="invalid_weights_model",
            description="Test numpy conversion error for model",
            model_weights=[[1,2]] # Data can be anything, as np.array is mocked
        )
        response = await upload_model(self.mock_ctx, args=request_args)

        self.assertIn("error", response)
        self.assertIn("Invalid model_weights data format for 'invalid_weights_model': Simulated np.array error for model weights", response["error"])
        mock_save_model.assert_not_called()
        self.mock_ctx.log_error.assert_any_call(
            "ValueError during model weights conversion for 'invalid_weights_model': Simulated np.array error for model weights",
            exc_info=True
        )

    @patch('tensordirectory.mcp_interface.storage.save_model', return_value=None)
    async def test_07c_upload_model_resource_storage_failure(self, mock_save_model):
        request_args = ModelUploadArgs( # Corrected variable name and Pydantic model name
            name="fail_model",
            description="Will fail",
            model_code="pass"
        )
        response = await upload_model(self.mock_ctx, args=request_args)

        self.assertIn("error", response)
        self.assertEqual(response["error"], "Failed to save model 'fail_model'")
        mock_save_model.assert_called_once()
        self.mock_ctx.log_error.assert_called_with("Failed to save model 'fail_model' - storage returned no UUID")

    # test_07d_upload_model_resource_missing_keys is removed as Pydantic handles this.

    @patch('tensordirectory.mcp_interface.invoke_agent_query', new_callable=AsyncMock)
    async def test_08_query_tensor_directory_tool(self, mock_invoke_agent_query):
        mock_agent_response = "Agent says hello!"
        mock_invoke_agent_query.return_value = mock_agent_response

        prompt_text = "What is tensor X?"
        params_dict = {"detail": "high"}

        tool_response = await query_tensor_directory(prompt=prompt_text, ctx=self.mock_ctx, params=params_dict)

        mock_invoke_agent_query.assert_called_once_with(prompt=prompt_text, params=params_dict, ctx=self.mock_ctx)
        self.assertEqual(tool_response, mock_agent_response)
        self.mock_ctx.log_info.assert_any_call(f"Query received for TensorDirectory: '{prompt_text}' with params: {params_dict}")

    # --- Tests for new utility tools ---

    # LIST TENSORS
    @patch('tensordirectory.mcp_interface.storage.list_tensors')
    async def test_09_list_tensors_success(self, mock_storage_list_tensors):
        mock_storage_list_tensors.return_value = (
            [
                {'uuid': 'u1', 'user_name': 't1', 'description': 'd1', 'creation_date': 'date1', 'original_dtype': 'float32', 'original_shape': '(10,)'},
                {'uuid': 'u2', 'user_name': 't2', 'description': 'd2', 'creation_date': 'date2', 'original_dtype': 'int32', 'original_shape': '(5,)'}
            ],
            2 # total_count
        )
        args = ListArgs(limit=5, offset=0)
        response = await list_tensors(self.mock_ctx, args=args)

        mock_storage_list_tensors.assert_called_once_with(filter_by_name_contains=None, limit=5, offset=0)
        self.assertIsInstance(response, ListTensorsResponse)
        self.assertEqual(len(response.tensors), 2)
        self.assertEqual(response.total_items_in_collection, 2)
        self.assertEqual(response.tensors[0].user_name, 't1')
        self.assertIsInstance(response.tensors[0], TensorMetadata)

    @patch('tensordirectory.mcp_interface.storage.list_tensors')
    async def test_10_list_tensors_empty(self, mock_storage_list_tensors):
        mock_storage_list_tensors.return_value = ([], 0)
        args = ListArgs()
        response = await list_tensors(self.mock_ctx, args=args)
        self.assertEqual(len(response.tensors), 0)
        self.assertEqual(response.total_items_in_collection, 0)

    @patch('tensordirectory.mcp_interface.storage.list_tensors', side_effect=Exception("Storage List Error"))
    async def test_11_list_tensors_storage_error(self, mock_storage_list_tensors_error):
        args = ListArgs()
        response = await list_tensors(self.mock_ctx, args=args)
        self.assertEqual(len(response.tensors), 0) # Should return empty list on error
        self.assertEqual(response.total_items_in_collection, 0)
        self.mock_ctx.log_error.assert_called_with("Error listing tensors: Storage List Error", exc_info=True)

    # LIST MODELS (similar structure to list_tensors)
    @patch('tensordirectory.mcp_interface.storage.list_models')
    async def test_12_list_models_success(self, mock_storage_list_models):
        mock_storage_list_models.return_value = (
            [
                {'uuid': 'm1', 'user_name': 'modelA', 'description': 'descA', 'upload_date': 'dateA', 'has_code': True, 'has_weights': False},
                {'uuid': 'm2', 'user_name': 'modelB', 'description': 'descB', 'upload_date': 'dateB', 'has_code': False, 'has_weights': True}
            ],
            2
        )
        args = ListArgs(filter_by_name_contains="model")
        response = await list_models(self.mock_ctx, args=args)
        mock_storage_list_models.assert_called_once_with(filter_by_name_contains="model", limit=args.limit, offset=args.offset)
        self.assertIsInstance(response, ListModelsResponse)
        self.assertEqual(len(response.models), 2)
        self.assertEqual(response.models[0].user_name, "modelA")
        self.assertTrue(response.models[0].has_code)
        self.assertIsInstance(response.models[0], ModelMetadata)

    # DELETE TENSOR
    @patch('tensordirectory.mcp_interface.storage.delete_tensor_by_uuid')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True) # Assume input is UUID
    async def test_13_delete_tensor_by_uuid_success(self, mock_is_uuid, mock_storage_delete):
        mock_storage_delete.return_value = True
        args = DeleteArgs(name_or_uuid="sample-uuid-to-delete")
        response = await delete_tensor(self.mock_ctx, args=args)

        mock_is_uuid.assert_called_once_with("sample-uuid-to-delete")
        mock_storage_delete.assert_called_once_with("sample-uuid-to-delete")
        self.assertEqual(response, {"success": True, "message": "Tensor 'sample-uuid-to-delete' (UUID: sample-uuid-to-delete) deleted successfully."})

    @patch('tensordirectory.mcp_interface.h5py.File')
    @patch('tensordirectory.mcp_interface.storage._get_uuid_from_name')
    @patch('tensordirectory.mcp_interface.storage.delete_tensor_by_uuid')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=False) # Assume input is a name
    async def test_14_delete_tensor_by_name_success(self, mock_is_uuid, mock_storage_delete, mock_get_uuid, mock_h5file):
        # Setup for _get_uuid_from_name call within the tool
        mock_hf_instance = MagicMock(spec=h5py.File)
        mock_h5file.return_value.__enter__.return_value = mock_hf_instance
        mock_get_uuid.return_value = "resolved-uuid-from-name"
        mock_storage_delete.return_value = True

        args = DeleteArgs(name_or_uuid="tensor_name_to_delete")
        response = await delete_tensor(self.mock_ctx, args=args)

        mock_is_uuid.assert_called_once_with("tensor_name_to_delete")
        mock_h5file.assert_called_once_with(tensordirectory.mcp_interface.storage.HDF5_FILE_NAME, 'r')
        mock_get_uuid.assert_called_once_with(mock_hf_instance, "tensor", "tensor_name_to_delete")
        mock_storage_delete.assert_called_once_with("resolved-uuid-from-name")
        self.assertTrue(response["success"])
        self.assertIn("deleted successfully", response["message"])

    @patch('tensordirectory.mcp_interface.h5py.File')
    @patch('tensordirectory.mcp_interface.storage._get_uuid_from_name')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=False) # Assume input is a name
    async def test_15_delete_tensor_name_not_found(self, mock_is_uuid, mock_get_uuid, mock_h5file):
        mock_hf_instance = MagicMock(spec=h5py.File)
        mock_h5file.return_value.__enter__.return_value = mock_hf_instance
        mock_get_uuid.return_value = None # Simulate name not found

        args = DeleteArgs(name_or_uuid="unknown_tensor_name")
        response = await delete_tensor(self.mock_ctx, args=args)

        self.assertEqual(response, {"success": False, "message": "Tensor 'unknown_tensor_name' not found by name."})

    # DELETE MODEL (similar structure to delete_tensor)
    @patch('tensordirectory.mcp_interface.storage.delete_model_by_uuid')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True)
    async def test_16_delete_model_by_uuid_success(self, mock_is_uuid, mock_storage_delete_model):
        mock_storage_delete_model.return_value = True
        args = DeleteArgs(name_or_uuid="model-uuid-to-delete")
        response = await delete_model(self.mock_ctx, args=args)
        mock_storage_delete_model.assert_called_once_with("model-uuid-to-delete")
        self.assertTrue(response["success"])

    # UPDATE TENSOR METADATA
    @patch('tensordirectory.mcp_interface.storage.update_tensor_metadata')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True) # Assume input is UUID
    async def test_17_update_tensor_metadata_success(self, mock_is_uuid, mock_storage_update):
        updated_metadata_payload = {"uuid": "uuid1", "user_name": "new_name", "description": "new_desc"}
        mock_storage_update.return_value = updated_metadata_payload

        args = UpdateMetadataArgs(name_or_uuid="uuid1", metadata_updates={"user_name": "new_name", "description": "new_desc"})
        response = await update_tensor_metadata(self.mock_ctx, args=args)

        mock_storage_update.assert_called_once_with("uuid1", {"user_name": "new_name", "description": "new_desc"})
        self.assertEqual(response, {"success": True, "metadata": updated_metadata_payload})

    @patch('tensordirectory.mcp_interface.storage.update_tensor_metadata')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True)
    async def test_18_update_tensor_metadata_item_not_found(self, mock_is_uuid, mock_storage_update):
        mock_storage_update.return_value = None # Simulate item not found by storage function
        args = UpdateMetadataArgs(name_or_uuid="uuid_not_exist", metadata_updates={"description": "new_desc"})
        response = await update_tensor_metadata(self.mock_ctx, args=args)
        self.assertEqual(response, {"success": False, "message": "Tensor UUID 'uuid_not_exist' not found or update failed."})

    # UPDATE MODEL METADATA (similar structure)
    @patch('tensordirectory.mcp_interface.storage.update_model_metadata')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True)
    async def test_19_update_model_metadata_success(self, mock_is_uuid, mock_storage_update_model):
        updated_model_meta = {"uuid": "muuid1", "user_name": "new_model_name", "has_code": True, "has_weights": False}
        mock_storage_update_model.return_value = updated_model_meta
        args = UpdateMetadataArgs(name_or_uuid="muuid1", metadata_updates={"user_name": "new_model_name"})
        response = await update_model_metadata(self.mock_ctx, args=args)
        mock_storage_update_model.assert_called_once_with("muuid1", {"user_name": "new_model_name"})
        self.assertEqual(response, {"success": True, "metadata": updated_model_meta})


if __name__ == '__main__':
    unittest.main(verbosity=2)
