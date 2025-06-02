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
    mcp_server # To check instance
)
from mcp.server.fastmcp import FastMCP, Context # For type checking and mock context

# It's often easier to mock the specific functions from storage that mcp_interface uses
# rather than the whole module.

class TestMCPInterface(unittest.IsolatedAsyncioTestCase):

    async def test_01_mcp_server_instance(self):
        self.assertIsInstance(mcp_server, FastMCP)

    @patch('tensordirectory.mcp_interface.storage.save_tensor', new_callable=MagicMock)
    async def test_02_upload_tensor_resource_success(self, mock_save_tensor):
        mock_uuid = "test-tensor-uuid-123"
        mock_save_tensor.return_value = mock_uuid
        
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock()
        mock_ctx.log_error = AsyncMock()

        tensor_list_data = [[1, 2], [3, 4]]
        response = await upload_tensor_resource(
            name="test_tensor",
            description="A test tensor",
            tensor_data=tensor_list_data,
            ctx=mock_ctx
        )

        mock_save_tensor.assert_called_once()
        args, kwargs = mock_save_tensor.call_args_list[0] # Get the first call's arguments
        # In this case, args is empty, kwargs contains the named arguments
        self.assertIsInstance(kwargs['tensor_data'], np.ndarray)
        np.testing.assert_array_equal(kwargs['tensor_data'], np.array(tensor_list_data))
        self.assertEqual(kwargs['name'], "test_tensor")
        self.assertEqual(kwargs['description'], "A test tensor")
        
        self.assertEqual(response, {"uuid": mock_uuid, "name": "test_tensor", "message": "Tensor uploaded successfully"})
        mock_ctx.log_info.assert_any_call("Attempting to upload tensor: test_tensor")
        mock_ctx.log_info.assert_any_call(f"Tensor 'test_tensor' saved with UUID: {mock_uuid}")


    @patch('tensordirectory.mcp_interface.storage.save_tensor', new_callable=MagicMock)
    async def test_03_upload_tensor_resource_invalid_data_empty(self, mock_save_tensor):
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock()
        mock_ctx.log_error = AsyncMock() # This might be called by the handler

        response = await upload_tensor_resource(name="invalid_tensor", description="Invalid", tensor_data=[], ctx=mock_ctx)
        self.assertIn("error", response)
        self.assertEqual(response["error"], "tensor_data must be a non-empty list")
        mock_save_tensor.assert_not_called()
        mock_ctx.log_info.assert_any_call("Attempting to upload tensor: invalid_tensor")
        # Depending on implementation, log_error might be called by the handler upon returning an error
        # For now, we don't assert log_error as the return itself indicates failure.

    @patch('tensordirectory.mcp_interface.storage.save_tensor', new_callable=MagicMock)
    async def test_03b_upload_tensor_resource_invalid_data_type(self, mock_save_tensor):
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock()
        mock_ctx.log_error = AsyncMock()

        response = await upload_tensor_resource(name="invalid_tensor_type", description="Invalid type", tensor_data="not a list", ctx=mock_ctx)
        self.assertIn("error", response)
        self.assertEqual(response["error"], "tensor_data must be a non-empty list") # Current check is basic
        mock_save_tensor.assert_not_called()


    @patch('tensordirectory.mcp_interface.storage.save_tensor', return_value=None) 
    async def test_04_upload_tensor_resource_storage_failure(self, mock_save_tensor_failure):
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock()
        mock_ctx.log_error = AsyncMock()

        response = await upload_tensor_resource(name="fail_tensor", description="Fail save", tensor_data=[[1]], ctx=mock_ctx)
        self.assertIn("error", response)
        self.assertEqual(response["error"], "Failed to save tensor 'fail_tensor'")
        mock_save_tensor_failure.assert_called_once()
        mock_ctx.log_error.assert_called_with("Failed to save tensor 'fail_tensor' - storage returned no UUID")


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_05_upload_model_resource_success_code_only(self, mock_save_model):
        mock_uuid = "test-model-uuid-456"
        mock_save_model.return_value = mock_uuid
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock()
        mock_ctx.log_error = AsyncMock()

        response = await upload_model_resource(
            name="code_model", description="Code only", model_code="print('hi')", model_weights=None, ctx=mock_ctx
        )
        mock_save_model.assert_called_once_with(name="code_model", description="Code only", model_weights=None, model_code="print('hi')")
        self.assertEqual(response, {"uuid": mock_uuid, "name": "code_model", "message": "Model uploaded successfully"})
        mock_ctx.log_info.assert_any_call("Attempting to upload model: code_model")
        mock_ctx.log_info.assert_any_call(f"Model 'code_model' saved with UUID: {mock_uuid}")


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_06_upload_model_resource_success_weights_only(self, mock_save_model):
        mock_uuid = "test-model-uuid-789"
        mock_save_model.return_value = mock_uuid
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock()
        mock_ctx.log_error = AsyncMock()

        weights_list = [[1.0, 2.0], [3.0, 4.0]]
        response = await upload_model_resource(
            name="weights_model", description="Weights only", model_weights=weights_list, model_code=None, ctx=mock_ctx
        )
        
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
        response = await upload_model_resource(
            name="cw_model", description="Code & Weights", model_weights=weights_list, model_code=code_str, ctx=mock_ctx
        )
        
        args, kwargs = mock_save_model.call_args_list[0]
        self.assertIsInstance(kwargs['model_weights'], np.ndarray)
        np.testing.assert_array_equal(kwargs['model_weights'], np.array(weights_list))
        self.assertEqual(kwargs['model_code'], code_str)
        self.assertEqual(kwargs['name'], "cw_model")
        self.assertEqual(response, {"uuid": mock_uuid, "name": "cw_model", "message": "Model uploaded successfully"})


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_07_upload_model_resource_no_code_no_weights(self, mock_save_model):
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock() 
        mock_ctx.log_error = AsyncMock()
        
        response = await upload_model_resource(name="empty_model", description="Empty", ctx=mock_ctx)
        self.assertIn("error", response)
        self.assertEqual(response["error"], "Either model_weights or model_code must be provided.")
        mock_save_model.assert_not_called()
        mock_ctx.log_info.assert_any_call("Attempting to upload model: empty_model")


    @patch('tensordirectory.mcp_interface.storage.save_model', new_callable=MagicMock)
    async def test_07b_upload_model_resource_invalid_weights_type(self, mock_save_model):
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock()
        mock_ctx.log_error = AsyncMock()

        response = await upload_model_resource(name="invalid_weights_model", description="Test", model_weights="not a list", ctx=mock_ctx)
        self.assertIn("error", response)
        self.assertEqual(response["error"], "model_weights must be a list if provided")
        mock_save_model.assert_not_called()

    @patch('tensordirectory.mcp_interface.storage.save_model', return_value=None)
    async def test_07c_upload_model_resource_storage_failure(self, mock_save_model):
        mock_ctx = MagicMock(spec=Context)
        mock_ctx.log_info = AsyncMock()
        mock_ctx.log_error = AsyncMock()

        response = await upload_model_resource(name="fail_model", description="Will fail", model_code="pass", ctx=mock_ctx)
        self.assertIn("error", response)
        self.assertEqual(response["error"], "Failed to save model 'fail_model'")
        mock_save_model.assert_called_once()
        mock_ctx.log_error.assert_called_with("Failed to save model 'fail_model' - storage returned no UUID")


    @patch('tensordirectory.mcp_interface.invoke_agent_query', new_callable=AsyncMock) 
    async def test_08_query_tensor_directory_tool(self, mock_invoke_agent_query):
        mock_agent_response = "Agent says hello!"
        mock_invoke_agent_query.return_value = mock_agent_response
        
        mock_ctx = MagicMock(spec=Context) 
        mock_ctx.log_info = AsyncMock() 

        prompt_text = "What is tensor X?"
        params_dict = {"detail": "high"}

        tool_response = await query_tensor_directory(prompt=prompt_text, params=params_dict, ctx=mock_ctx)

        mock_invoke_agent_query.assert_called_once_with(prompt_text, params_dict, mock_ctx)
        self.assertEqual(tool_response, mock_agent_response)
        mock_ctx.log_info.assert_any_call(f"Query received for TensorDirectory: '{prompt_text}' with params: {params_dict}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
