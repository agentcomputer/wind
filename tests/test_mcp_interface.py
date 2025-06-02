# tests/test_mcp_interface.py
import unittest
from unittest.mock import patch, AsyncMock, MagicMock 
# AsyncMock for async functions, MagicMock for synchronous or context
import numpy as np
import asyncio # Required for IsolatedAsyncioTestCase if not using it directly

# Assuming tensordirectory is in PYTHONPATH or project is structured as a package
from tensordirectory.mcp_interface import (
    upload_tensor_resource,
    upload_model_resource,
    query_tensor_directory,
    mcp_server, # To check instance
    TensorUploadRequest, # Import Pydantic model
    ModelUploadRequest   # Import Pydantic model
)
from mcp.server.fastmcp import FastMCP, Context # For type checking and mock context

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
        request_data = TensorUploadRequest(
            name="test_tensor",
            description="A test tensor",
            tensor_data=tensor_list_data
        )
        response = await upload_tensor_resource(self.mock_ctx, data=request_data)

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
        request_data = TensorUploadRequest(name="invalid_tensor", description="Invalid", tensor_data=[])
        response = await upload_tensor_resource(self.mock_ctx, data=request_data)
        
        self.assertIn("error", response)
        # The error message comes from within the handler after Pydantic validation passes for `list` type
        self.assertEqual(response["error"], "tensor_data for 'invalid_tensor' must be a non-empty list.")
        mock_save_tensor.assert_not_called()
        self.mock_ctx.log_info.assert_any_call("Attempting to upload tensor: invalid_tensor")
        self.mock_ctx.log_error.assert_called_with("tensor_data for 'invalid_tensor' must be a non-empty list.")

    @patch('tensordirectory.mcp_interface.storage.save_tensor', new_callable=MagicMock)
    async def test_03b_upload_tensor_resource_value_error_on_conversion(self, mock_save_tensor):
        # This test checks if np.array() fails inside the handler
        # Pydantic will allow tensor_data=[["a", "b"], [1,2]] because it's a list of lists.
        request_data = TensorUploadRequest(
            name="value_error_tensor", 
            description="Numpy conversion failure test", 
            tensor_data=[["a", "b"], [1,2]] # This will cause np.array() to likely raise ValueError
        )
        # Mock save_tensor to do nothing as it won't be reached if np.array fails
        # or if it's reached, we don't care about its return for this test.
        
        response = await upload_tensor_resource(self.mock_ctx, data=request_data)
        
        self.assertIn("error", response)
        self.assertTrue(response["error"].startswith("Invalid tensor data format for 'value_error_tensor':"))
        mock_save_tensor.assert_not_called() # Should fail before saving
        self.mock_ctx.log_error.assert_any_call(
            "ValueError during tensor conversion for 'value_error_tensor': object of too small depth for desired array", 
            exc_info=True
        ) # Check for the specific log, actual error message from np might vary.

    # test_03c_upload_tensor_resource_missing_keys is removed as Pydantic handles this.
    # FastMCP would typically return a 422 error if Pydantic validation fails.

    @patch('tensordirectory.mcp_interface.storage.save_tensor', return_value=None)
    async def test_04_upload_tensor_resource_storage_failure(self, mock_save_tensor_failure):
        request_data = TensorUploadRequest(name="fail_tensor", description="Fail save", tensor_data=[[1]])
        response = await upload_tensor_resource(self.mock_ctx, data=request_data)
        
        self.assertIn("error", response)
        self.assertEqual(response["error"], "Failed to save tensor 'fail_tensor'")
        mock_save_tensor_failure.assert_called_once()
        self.mock_ctx.log_error.assert_called_with("Failed to save tensor 'fail_tensor' - storage returned no UUID")


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_05_upload_model_resource_success_code_only(self, mock_save_model):
        mock_uuid = "test-model-uuid-456"
        mock_save_model.return_value = mock_uuid
        
        request_data = ModelUploadRequest(
            name="code_model", 
            description="Code only", 
            model_code="print('hi')"
        )
        response = await upload_model_resource(self.mock_ctx, data=request_data)
        
        mock_save_model.assert_called_once_with(name="code_model", description="Code only", model_weights=None, model_code="print('hi')")
        self.assertEqual(response, {"uuid": mock_uuid, "name": "code_model", "message": "Model uploaded successfully"})
        self.mock_ctx.log_info.assert_any_call("Attempting to upload model: code_model")
        self.mock_ctx.log_info.assert_any_call(f"Model 'code_model' saved with UUID: {mock_uuid}")


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_06_upload_model_resource_success_weights_only(self, mock_save_model):
        mock_uuid = "test-model-uuid-789"
        mock_save_model.return_value = mock_uuid
        
        weights_list = [[1.0, 2.0], [3.0, 4.0]]
        request_data = ModelUploadRequest(
            name="weights_model", 
            description="Weights only", 
            model_weights=weights_list
        )
        response = await upload_model_resource(self.mock_ctx, data=request_data)
        
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
        request_data = ModelUploadRequest(
            name="cw_model", 
            description="Code & Weights", 
            model_weights=weights_list_for_cw, 
            model_code=code_str
        )
        response = await upload_model_resource(self.mock_ctx, data=request_data)
        
        args, kwargs = mock_save_model.call_args_list[0]
        self.assertIsInstance(kwargs['model_weights'], np.ndarray)
        np.testing.assert_array_equal(kwargs['model_weights'], np.array(weights_list))
        self.assertEqual(kwargs['model_code'], code_str)
        self.assertEqual(kwargs['name'], "cw_model")
        self.assertEqual(response, {"uuid": mock_uuid, "name": "cw_model", "message": "Model uploaded successfully"})


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_07_upload_model_resource_no_code_no_weights(self, mock_save_model):
        # self.mock_ctx is used from setUp
        request_data = ModelUploadRequest(name="empty_model", description="Empty") # No model_code or model_weights
        response = await upload_model_resource(self.mock_ctx, data=request_data)
        
        self.assertIn("error", response)
        self.assertEqual(response["error"], "Either model_weights or model_code must be provided.")
        mock_save_model.assert_not_called()
        self.mock_ctx.log_info.assert_any_call("Attempting to upload model: empty_model")
        self.mock_ctx.log_error.assert_called_with("Validation error for model 'empty_model': Either model_weights or model_code must be provided.")


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_07b_upload_model_resource_value_error_on_conversion(self, mock_save_model):
        # Pydantic ensures model_weights is a list if provided.
        # This test is for when np.array(list_data) fails.
        request_data = ModelUploadRequest(
            name="invalid_weights_model", 
            description="Test numpy conversion error", 
            model_weights=[["a", "b"], [1,2]] # This should cause np.array to fail
        )
        response = await upload_model_resource(self.mock_ctx, data=request_data)

        self.assertIn("error", response)
        self.assertTrue(response["error"].startswith("Invalid model_weights data format for 'invalid_weights_model':"))
        mock_save_model.assert_not_called()
        self.mock_ctx.log_error.assert_any_call(
            "ValueError during model weights conversion for 'invalid_weights_model': object of too small depth for desired array",
            exc_info=True
        )

    @patch('tensordirectory.mcp_interface.storage.save_model', return_value=None)
    async def test_07c_upload_model_resource_storage_failure(self, mock_save_model):
        request_data = ModelUploadRequest(
            name="fail_model", 
            description="Will fail", 
            model_code="pass"
        )
        response = await upload_model_resource(self.mock_ctx, data=request_data)

        self.assertIn("error", response)
        self.assertEqual(response["error"], "Failed to save model 'fail_model'")
        mock_save_model.assert_called_once()
        self.mock_ctx.log_error.assert_called_with("Failed to save model 'fail_model' - storage returned no UUID")

    # test_07d_upload_model_resource_missing_keys is removed as Pydantic handles this.

    @patch('tensordirectory.mcp_interface.invoke_agent_query', new_callable=AsyncMock)
    async def test_08_query_tensor_directory_tool(self, mock_invoke_agent_query):
        mock_agent_response = "Agent says hello!"
        mock_invoke_agent_query.return_value = mock_agent_response
        
        # self.mock_ctx is already set up in setUp()
        prompt_text = "What is tensor X?"
        params_dict = {"detail": "high"}

        tool_response = await query_tensor_directory(prompt=prompt_text, ctx=self.mock_ctx, params=params_dict)

        mock_invoke_agent_query.assert_called_once_with(prompt_text, params_dict, self.mock_ctx)
        self.assertEqual(tool_response, mock_agent_response)
        self.mock_ctx.log_info.assert_any_call(f"Query received for TensorDirectory: '{prompt_text}' with params: {params_dict}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
