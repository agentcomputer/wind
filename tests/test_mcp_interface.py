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
import os # For file operations
import tempfile # For temporary file names
from tensordirectory import storage as actual_storage # To call _initialize_hdf5 and save functions directly

# Mock google.generativeai at the module level to avoid import errors if agent uses it
# This needs to be done before 'from tensordirectory.mcp_interface import ...' if agent is imported by mcp_interface
# However, mcp_interface itself doesn't directly import agent in a way that triggers genai import at load time
# for all tools. Let's put it here for safety, or it could be a fixture in pytest.
# For unittest, it's common to patch it in setUpModule or at class level if needed.
# Simpler: add to sys.modules
import sys
mock_genai = MagicMock()
sys.modules['google.generativeai'] = mock_genai
sys.modules['google.ai'] = MagicMock() # If 'google.ai.generativelanguage' is also imported by the agent


# Store the original HDF5 file name
ORIGINAL_HDF5_FILE_NAME = actual_storage.HDF5_FILE_NAME
TEMP_HDF5_FILE_NAME = "" # Will be set in setUp

class TestMCPInterfaceWithRealStorage(unittest.IsolatedAsyncioTestCase):
    """
    Tests MCP interface tools against a real (temporary) HDF5 storage.
    """
    @classmethod
    def setUpClass(cls):
        global TEMP_HDF5_FILE_NAME
        # Create a temporary file name for HDF5 storage for the whole test class
        # This ensures all tests in this class use the same temp file,
        # allowing for state to persist across tests if needed (e.g. create then list)
        # or be cleaned up methodically. For true isolation, setUp/tearDown per method is better.
        # Let's try per-method isolation for cleaner tests.
        pass


    @classmethod
    def tearDownClass(cls):
        pass # Cleanup handled per method if needed

    def setUp(self):
        # Common mock context for all tests in this class
        self.mock_ctx = MagicMock(spec=Context)
        self.mock_ctx.log_info = AsyncMock()
        self.mock_ctx.log_error = AsyncMock()
        self.mock_ctx.log_warning = AsyncMock()

        # Create a new temporary HDF5 file for each test method
        # This provides test isolation.
        temp_file = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        self.temp_hdf5_path = temp_file.name
        temp_file.close() # Close it so h5py can open it exclusively

        actual_storage.HDF5_FILE_NAME = self.temp_hdf5_path
        actual_storage._initialize_hdf5() # Initialize the temporary HDF5 file

        # If the agent or its dependencies are complex to mock for all tools,
        # and only query_tensor_directory uses it, we could patch invoke_agent_query
        # for tests not focusing on the agent.
        # For now, the google.generativeai mock should prevent crashes.

    def tearDown(self):
        # Ensure the original HDF5_FILE_NAME is restored
        actual_storage.HDF5_FILE_NAME = ORIGINAL_HDF5_FILE_NAME
        # Clean up the temporary HDF5 file
        if os.path.exists(self.temp_hdf5_path):
            os.remove(self.temp_hdf5_path)

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
    # These tests will use the real storage backend with a temporary HDF5 file.

    async def helper_upload_tensor(self, name: str, desc: str, data: list) -> dict:
        """Helper to upload a tensor using the MCP tool."""
        args = TensorUploadArgs(name=name, description=desc, tensor_data=data)
        return await upload_tensor(self.mock_ctx, args=args)

    async def helper_upload_model(self, name: str, desc: str, code: str = None, weights: list = None) -> dict:
        """Helper to upload a model using the MCP tool."""
        args = ModelUploadArgs(name=name, description=desc, model_code=code, model_weights=weights)
        return await upload_model(self.mock_ctx, args=args)

    # LIST TENSORS
    async def test_20_list_tensors_empty(self):
        args = ListArgs()
        response = await list_tensors(self.mock_ctx, args=args)
        self.assertIsInstance(response, ListTensorsResponse)
        self.assertEqual(len(response.tensors), 0)
        self.assertEqual(response.total_items_in_collection, 0)

    async def test_21_list_tensors_with_items_and_filter_pagination(self):
        await self.helper_upload_tensor("tensor_apple_1", "desc1", [[1,0]])
        t2_resp = await self.helper_upload_tensor("tensor_banana_2", "desc2", [[2,0]])
        await self.helper_upload_tensor("tensor_apple_3", "desc3", [[3,0]])

        # List all
        response_all = await list_tensors(self.mock_ctx, args=ListArgs())
        self.assertEqual(response_all.total_items_in_collection, 3)
        self.assertEqual(len(response_all.tensors), 3) # Default limit is 100

        # Filter by name
        response_apple = await list_tensors(self.mock_ctx, args=ListArgs(filter_by_name_contains="apple"))
        self.assertEqual(response_apple.total_items_in_collection, 2)
        self.assertEqual(len(response_apple.tensors), 2)
        for t in response_apple.tensors:
            self.assertIn("apple", t.user_name)

        response_banana = await list_tensors(self.mock_ctx, args=ListArgs(filter_by_name_contains="banana"))
        self.assertEqual(response_banana.total_items_in_collection, 1)
        self.assertEqual(len(response_banana.tensors), 1)
        self.assertEqual(response_banana.tensors[0].user_name, "tensor_banana_2")
        self.assertEqual(response_banana.tensors[0].uuid, t2_resp["uuid"])


        # Test pagination: limit
        response_limit1 = await list_tensors(self.mock_ctx, args=ListArgs(limit=1))
        self.assertEqual(response_limit1.total_items_in_collection, 3)
        self.assertEqual(len(response_limit1.tensors), 1)

        # Test pagination: offset (names are sorted: apple_1, apple_3, banana_2)
        # actual_storage.list_tensors sorts by name by default.
        # So, order should be apple_1, apple_3, banana_2
        all_tensors_sorted = sorted([r.user_name for r in response_all.tensors])


        response_offset1_limit1 = await list_tensors(self.mock_ctx, args=ListArgs(limit=1, offset=1))
        self.assertEqual(len(response_offset1_limit1.tensors), 1)
        self.assertEqual(response_offset1_limit1.tensors[0].user_name, all_tensors_sorted[1])


        response_offset2_limit2 = await list_tensors(self.mock_ctx, args=ListArgs(limit=2, offset=2))
        self.assertEqual(len(response_offset2_limit2.tensors), 1) # Only one left
        self.assertEqual(response_offset2_limit2.tensors[0].user_name, all_tensors_sorted[2])

    # LIST MODELS
    async def test_22_list_models_empty(self):
        args = ListArgs()
        response = await list_models(self.mock_ctx, args=args)
        self.assertIsInstance(response, ListModelsResponse)
        self.assertEqual(len(response.models), 0)
        self.assertEqual(response.total_items_in_collection, 0)

    async def test_23_list_models_with_items_and_filter_pagination(self):
        await self.helper_upload_model("model_cat_1", "desc1_cat", code="c1")
        m2_resp = await self.helper_upload_model("model_dog_2", "desc2_dog", code="c2")
        await self.helper_upload_model("model_cat_3", "desc3_cat", code="c3")

        response_all = await list_models(self.mock_ctx, args=ListArgs())
        self.assertEqual(response_all.total_items_in_collection, 3)
        self.assertEqual(len(response_all.models), 3)

        response_cat = await list_models(self.mock_ctx, args=ListArgs(filter_by_name_contains="cat"))
        self.assertEqual(response_cat.total_items_in_collection, 2)
        self.assertEqual(len(response_cat.models), 2)

        response_dog = await list_models(self.mock_ctx, args=ListArgs(filter_by_name_contains="dog"))
        self.assertEqual(response_dog.total_items_in_collection, 1)
        self.assertEqual(response_dog.models[0].user_name, "model_dog_2")
        self.assertEqual(response_dog.models[0].uuid, m2_resp["uuid"])

        # Test pagination (names sorted: cat_1, cat_3, dog_2)
        all_models_sorted = sorted([r.user_name for r in response_all.models])

        response_offset1_limit1 = await list_models(self.mock_ctx, args=ListArgs(limit=1, offset=1))
        self.assertEqual(len(response_offset1_limit1.models), 1)
        self.assertEqual(response_offset1_limit1.models[0].user_name, all_models_sorted[1])


    # DELETE TENSOR
    async def test_24_delete_tensor_by_uuid_and_name(self):
        # Add a tensor
        t1_upload = await self.helper_upload_tensor("del_tensor_1", "desc_del_1", [[1]])
        t1_uuid = t1_upload["uuid"]
        t1_name = t1_upload["name"]

        t2_upload = await self.helper_upload_tensor("del_tensor_2", "desc_del_2", [[2]])
        t2_uuid = t2_upload["uuid"]
        t2_name = t2_upload["name"]

        # Delete t1 by UUID
        del_t1_resp = await delete_tensor(self.mock_ctx, args=DeleteArgs(name_or_uuid=t1_uuid))
        self.assertTrue(del_t1_resp["success"])
        self.assertIn(t1_uuid, del_t1_resp["message"])

        # Verify t1 is deleted
        list_after_del_t1 = await list_tensors(self.mock_ctx, args=ListArgs(filter_by_name_contains=t1_name))
        self.assertEqual(list_after_del_t1.total_items_in_collection, 0)
        # Try deleting already deleted UUID
        del_t1_again_resp = await delete_tensor(self.mock_ctx, args=DeleteArgs(name_or_uuid=t1_uuid))
        self.assertFalse(del_t1_again_resp["success"]) # Should fail as it's gone from storage
        self.assertIn("not found or delete failed", del_t1_again_resp["message"])


        # Delete t2 by name
        del_t2_resp = await delete_tensor(self.mock_ctx, args=DeleteArgs(name_or_uuid=t2_name))
        self.assertTrue(del_t2_resp["success"])
        self.assertIn(t2_name, del_t2_resp["message"])
        self.assertIn(t2_uuid, del_t2_resp["message"]) # Message contains resolved UUID

        # Verify t2 is deleted
        list_after_del_t2 = await list_tensors(self.mock_ctx, args=ListArgs(filter_by_name_contains=t2_name))
        self.assertEqual(list_after_del_t2.total_items_in_collection, 0)

    async def test_25_delete_tensor_non_existent(self):
        # By UUID
        del_resp_uuid = await delete_tensor(self.mock_ctx, args=DeleteArgs(name_or_uuid="non-existent-uuid"))
        self.assertFalse(del_resp_uuid["success"])
        self.assertIn("not found or delete failed", del_resp_uuid["message"])

        # By name
        del_resp_name = await delete_tensor(self.mock_ctx, args=DeleteArgs(name_or_uuid="non_existent_name"))
        self.assertFalse(del_resp_name["success"])
        self.assertIn("not found by name", del_resp_name["message"])

    # DELETE MODEL
    async def test_26_delete_model_by_uuid_and_name(self):
        m1_upload = await self.helper_upload_model("del_model_1", "desc_del_m1", code="c1")
        m1_uuid = m1_upload["uuid"]
        m1_name = m1_upload["name"]

        m2_upload = await self.helper_upload_model("del_model_2", "desc_del_m2", code="c2")
        m2_uuid = m2_upload["uuid"]
        m2_name = m2_upload["name"]

        # Delete m1 by UUID
        del_m1_resp = await delete_model(self.mock_ctx, args=DeleteArgs(name_or_uuid=m1_uuid))
        self.assertTrue(del_m1_resp["success"])

        # Delete m2 by name
        del_m2_resp = await delete_model(self.mock_ctx, args=DeleteArgs(name_or_uuid=m2_name))
        self.assertTrue(del_m2_resp["success"])

        # Verify deletions
        list_all = await list_models(self.mock_ctx, args=ListArgs())
        self.assertEqual(list_all.total_items_in_collection, 0)

    # UPDATE TENSOR METADATA
    async def test_27_update_tensor_metadata_by_uuid_and_name(self):
        t_upload = await self.helper_upload_tensor("update_me_tensor", "old_desc", [[1]])
        t_uuid = t_upload["uuid"]
        original_name = t_upload["name"]

        # Update by UUID
        update_1_payload = {"description": "new_desc_uuid", "user_name": "updated_name_by_uuid"}
        resp1 = await update_tensor_metadata(self.mock_ctx, args=UpdateMetadataArgs(name_or_uuid=t_uuid, metadata_updates=update_1_payload))
        self.assertTrue(resp1["success"])
        self.assertEqual(resp1["metadata"]["description"], "new_desc_uuid")
        self.assertEqual(resp1["metadata"]["user_name"], "updated_name_by_uuid")
        self.assertEqual(resp1["metadata"]["uuid"], t_uuid) # Ensure UUID is still there

        # Verify old name does not work for listing, new name does
        list_old_name = await list_tensors(self.mock_ctx, args=ListArgs(filter_by_name_contains=original_name))
        self.assertEqual(list_old_name.total_items_in_collection, 0)
        list_new_name = await list_tensors(self.mock_ctx, args=ListArgs(filter_by_name_contains="updated_name_by_uuid"))
        self.assertEqual(list_new_name.total_items_in_collection, 1)


        # Update by new name
        current_name = "updated_name_by_uuid"
        update_2_payload = {"description": "final_desc_name", "user_name": "final_tensor_name"}
        resp2 = await update_tensor_metadata(self.mock_ctx, args=UpdateMetadataArgs(name_or_uuid=current_name, metadata_updates=update_2_payload))
        self.assertTrue(resp2["success"])
        self.assertEqual(resp2["metadata"]["description"], "final_desc_name")
        self.assertEqual(resp2["metadata"]["user_name"], "final_tensor_name")

        # Verify creation_date, original_shape, original_dtype are not changed (they are not in metadata_updates)
        # And that they exist
        self.assertIn("creation_date", resp2["metadata"])
        self.assertIsNotNone(resp2["metadata"]["creation_date"])
        self.assertEqual(resp2["metadata"]["original_shape"], "str((1, 1))") # Shape from [[1]]
        self.assertEqual(resp2["metadata"]["original_dtype"], str(np.array([[1]]).dtype))


    async def test_28_update_tensor_metadata_non_existent(self):
        resp_uuid = await update_tensor_metadata(self.mock_ctx, args=UpdateMetadataArgs(name_or_uuid="non-existent-uuid", metadata_updates={"desc": "d"}))
        self.assertFalse(resp_uuid["success"])
        self.assertIn("not found or update failed", resp_uuid["message"])

        resp_name = await update_tensor_metadata(self.mock_ctx, args=UpdateMetadataArgs(name_or_uuid="non-existent-name", metadata_updates={"desc": "d"}))
        self.assertFalse(resp_name["success"])
        self.assertIn("not found by name", resp_name["message"])

    async def test_29_update_tensor_metadata_immutable_fields(self):
        # Attempt to update immutable fields like uuid, creation_date
        # The current storage.update_tensor_metadata only allows 'user_name' and 'description'.
        # Other fields in metadata_updates are ignored.
        t_upload = await self.helper_upload_tensor("immutable_test_tensor", "desc_imm", [[1]])
        t_uuid = t_upload["uuid"]
        t_metadata_before = await list_tensors(self.mock_ctx, args=ListArgs(filter_by_name_contains="immutable_test_tensor"))
        original_creation_date = t_metadata_before.tensors[0].creation_date

        update_payload = {"uuid": "fake-new-uuid", "creation_date": "fake-new-date", "description": "immutable_desc_updated"}
        resp = await update_tensor_metadata(self.mock_ctx, args=UpdateMetadataArgs(name_or_uuid=t_uuid, metadata_updates=update_payload))

        self.assertTrue(resp["success"])
        self.assertEqual(resp["metadata"]["uuid"], t_uuid) # Should not change
        self.assertEqual(resp["metadata"]["creation_date"], original_creation_date) # Should not change
        self.assertEqual(resp["metadata"]["description"], "immutable_desc_updated") # This should change

    # UPDATE MODEL METADATA
    async def test_30_update_model_metadata_by_uuid_and_name(self):
        m_upload = await self.helper_upload_model("update_me_model", "old_model_desc", code="c")
        m_uuid = m_upload["uuid"]
        original_model_name = m_upload["name"]

        # Update by UUID
        update_1_payload = {"description": "new_model_desc_uuid", "user_name": "updated_model_name_uuid"}
        resp1 = await update_model_metadata(self.mock_ctx, args=UpdateMetadataArgs(name_or_uuid=m_uuid, metadata_updates=update_1_payload))
        self.assertTrue(resp1["success"])
        self.assertEqual(resp1["metadata"]["description"], "new_model_desc_uuid")
        self.assertEqual(resp1["metadata"]["user_name"], "updated_model_name_uuid")
        self.assertTrue(resp1["metadata"]["has_code"]) # Check other fields remain

        # Verify old name fails, new name works
        list_old_name = await list_models(self.mock_ctx, args=ListArgs(filter_by_name_contains=original_model_name))
        self.assertEqual(list_old_name.total_items_in_collection, 0)
        list_new_name = await list_models(self.mock_ctx, args=ListArgs(filter_by_name_contains="updated_model_name_uuid"))
        self.assertEqual(list_new_name.total_items_in_collection, 1)

        # Update by new name
        current_name = "updated_model_name_uuid"
        update_2_payload = {"description": "final_model_desc_name", "user_name": "final_model_name"}
        resp2 = await update_model_metadata(self.mock_ctx, args=UpdateMetadataArgs(name_or_uuid=current_name, metadata_updates=update_2_payload))
        self.assertTrue(resp2["success"])
        self.assertEqual(resp2["metadata"]["user_name"], "final_model_name")
        self.assertIn("upload_date", resp2["metadata"]) # Ensure original other fields are there
        self.assertIsNotNone(resp2["metadata"]["upload_date"])


class TestMCPInterfaceMockedStorage(unittest.IsolatedAsyncioTestCase): # Renamed original class
    # This class keeps the original mock-based tests.
    # I'll copy the original setUp and tests 01-19 here.
    # For brevity in this example, I will only copy setUp and one test.
    # In a real scenario, all original tests (test_01_mcp_server_instance to test_19_update_model_metadata_success)
    # would be under this class.

    def setUp(self):
        # Common mock context for all tests in this class
        self.mock_ctx = MagicMock(spec=Context)
        self.mock_ctx.log_info = AsyncMock()
        self.mock_ctx.log_error = AsyncMock()
        self.mock_ctx.log_warning = AsyncMock() # If any handler uses log_warning

    async def test_01_mcp_server_instance(self): # Original test_01
        self.assertIsInstance(mcp_server, FastMCP)

    @patch('tensordirectory.mcp_interface.storage.save_tensor', new_callable=MagicMock)
    async def test_02_upload_tensor_resource_success(self, mock_save_tensor): # Original test_02
        mock_uuid = "test-tensor-uuid-123"
        mock_save_tensor.return_value = mock_uuid

        tensor_list_data = [[1, 2], [3, 4]]
        request_args = TensorUploadArgs(
            name="test_tensor",
            description="A test tensor",
            tensor_data=tensor_list_data
        )
        response = await upload_tensor(self.mock_ctx, args=request_args)

        mock_save_tensor.assert_called_once()
        args, kwargs = mock_save_tensor.call_args_list[0]
        self.assertIsInstance(kwargs['tensor_data'], np.ndarray)
        np.testing.assert_array_equal(kwargs['tensor_data'], np.array(tensor_list_data))
        self.assertEqual(kwargs['name'], "test_tensor")
        self.assertEqual(kwargs['description'], "A test tensor")

        self.assertEqual(response, {"uuid": mock_uuid, "name": "test_tensor", "message": "Tensor uploaded successfully"})
        self.mock_ctx.log_info.assert_any_call("Attempting to upload tensor: test_tensor")
        self.mock_ctx.log_info.assert_any_call(f"Tensor 'test_tensor' saved with UUID: {mock_uuid}")

    # ... (Original tests 03 through 08 would be here, unchanged) ...

    # --- Original Mocked Tests for new utility tools ---
    # These are the tests (09-19) that were already present but used mocks.
    # They are kept under this separate class.

    @patch('tensordirectory.mcp_interface.storage.list_tensors')
    async def test_09_list_tensors_success(self, mock_storage_list_tensors): # Original test_09
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
    async def test_10_list_tensors_empty(self, mock_storage_list_tensors): # Original test_10
        mock_storage_list_tensors.return_value = ([], 0)
        args = ListArgs()
        response = await list_tensors(self.mock_ctx, args=args)
        self.assertEqual(len(response.tensors), 0)
        self.assertEqual(response.total_items_in_collection, 0)

    @patch('tensordirectory.mcp_interface.storage.list_tensors', side_effect=Exception("Storage List Error"))
    async def test_11_list_tensors_storage_error(self, mock_storage_list_tensors_error): # Original test_11
        args = ListArgs()
        response = await list_tensors(self.mock_ctx, args=args)
        self.assertEqual(len(response.tensors), 0)
        self.assertEqual(response.total_items_in_collection, 0)
        self.mock_ctx.log_error.assert_called_with("Error listing tensors: Storage List Error", exc_info=True)

    @patch('tensordirectory.mcp_interface.storage.list_models')
    async def test_12_list_models_success(self, mock_storage_list_models): # Original test_12
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

    @patch('tensordirectory.mcp_interface.storage.delete_tensor_by_uuid')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True)
    async def test_13_delete_tensor_by_uuid_success(self, mock_is_uuid, mock_storage_delete): # Original test_13
        mock_storage_delete.return_value = True
        args = DeleteArgs(name_or_uuid="sample-uuid-to-delete")
        response = await delete_tensor(self.mock_ctx, args=args)

        mock_is_uuid.assert_called_once_with("sample-uuid-to-delete")
        mock_storage_delete.assert_called_once_with("sample-uuid-to-delete")
        self.assertEqual(response, {"success": True, "message": "Tensor 'sample-uuid-to-delete' (UUID: sample-uuid-to-delete) deleted successfully."})

    @patch('tensordirectory.mcp_interface.h5py.File')
    @patch('tensordirectory.mcp_interface.storage._get_uuid_from_name')
    @patch('tensordirectory.mcp_interface.storage.delete_tensor_by_uuid')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=False)
    async def test_14_delete_tensor_by_name_success(self, mock_is_uuid, mock_storage_delete, mock_get_uuid, mock_h5file): # Original test_14
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
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=False)
    async def test_15_delete_tensor_name_not_found(self, mock_is_uuid, mock_get_uuid, mock_h5file): # Original test_15
        mock_hf_instance = MagicMock(spec=h5py.File)
        mock_h5file.return_value.__enter__.return_value = mock_hf_instance
        mock_get_uuid.return_value = None

        args = DeleteArgs(name_or_uuid="unknown_tensor_name")
        response = await delete_tensor(self.mock_ctx, args=args)

        self.assertEqual(response, {"success": False, "message": "Tensor 'unknown_tensor_name' not found by name."})

    @patch('tensordirectory.mcp_interface.storage.delete_model_by_uuid')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True)
    async def test_16_delete_model_by_uuid_success(self, mock_is_uuid, mock_storage_delete_model): # Original test_16
        mock_storage_delete_model.return_value = True
        args = DeleteArgs(name_or_uuid="model-uuid-to-delete")
        response = await delete_model(self.mock_ctx, args=args)
        mock_storage_delete_model.assert_called_once_with("model-uuid-to-delete")
        self.assertTrue(response["success"])

    @patch('tensordirectory.mcp_interface.storage.update_tensor_metadata')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True)
    async def test_17_update_tensor_metadata_success(self, mock_is_uuid, mock_storage_update): # Original test_17
        updated_metadata_payload = {"uuid": "uuid1", "user_name": "new_name", "description": "new_desc"}
        mock_storage_update.return_value = updated_metadata_payload

        args = UpdateMetadataArgs(name_or_uuid="uuid1", metadata_updates={"user_name": "new_name", "description": "new_desc"})
        response = await update_tensor_metadata(self.mock_ctx, args=args)

        mock_storage_update.assert_called_once_with("uuid1", {"user_name": "new_name", "description": "new_desc"})
        self.assertEqual(response, {"success": True, "metadata": updated_metadata_payload})

    @patch('tensordirectory.mcp_interface.storage.update_tensor_metadata')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True)
    async def test_18_update_tensor_metadata_item_not_found(self, mock_is_uuid, mock_storage_update): # Original test_18
        mock_storage_update.return_value = None
        args = UpdateMetadataArgs(name_or_uuid="uuid_not_exist", metadata_updates={"description": "new_desc"})
        response = await update_tensor_metadata(self.mock_ctx, args=args)
        self.assertEqual(response, {"success": False, "message": "Tensor UUID 'uuid_not_exist' not found or update failed."})

    @patch('tensordirectory.mcp_interface.storage.update_model_metadata')
    @patch('tensordirectory.mcp_interface._is_uuid', return_value=True)
    async def test_19_update_model_metadata_success(self, mock_is_uuid, mock_storage_update_model): # Original test_19
        updated_model_meta = {"uuid": "muuid1", "user_name": "new_model_name", "has_code": True, "has_weights": False}
        mock_storage_update_model.return_value = updated_model_meta
        args = UpdateMetadataArgs(name_or_uuid="muuid1", metadata_updates={"user_name": "new_model_name"})
        response = await update_model_metadata(self.mock_ctx, args=args)
        mock_storage_update_model.assert_called_once_with("muuid1", {"user_name": "new_model_name"})
        self.assertEqual(response, {"success": True, "metadata": updated_model_meta})


if __name__ == '__main__':
    # Note: Running this directly will run both test classes.
    # Pytest discovery would also pick up both.
    unittest.main(verbosity=2)
