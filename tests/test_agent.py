# tests/test_agent.py
import unittest
from unittest.mock import patch, AsyncMock, MagicMock, PropertyMock
import os
import json
import numpy as np

# Assuming tensordirectory is in PYTHONPATH or project is structured as a package
from tensordirectory import agent
from tensordirectory.agent import analyze_prompt_with_gemini, handle_query, ensure_gemini_configured
from mcp.server.fastmcp import Context # For type hinting and mock context

# Mock the google.generativeai library structure that agent.py uses
# This is a simplified mock for the Gemini API response object
class MockGeminiResponse:
    def __init__(self, text_content="", parts=None):
        self.text_content = text_content
        self._parts = parts if parts is not None else []
        # Simulate the .text property behavior based on parts if text_content is not directly provided
        if not self.text_content and self._parts:
             self.text_content = "".join(part.text for part in self._parts if hasattr(part, 'text'))


    @property
    def text(self):
        return self.text_content
    
    @property
    def parts(self): # Make parts accessible
        return self._parts


class MockGenerativeModel:
    def __init__(self, model_name='gemini-pro'):
        self.model_name = model_name
        # This method will be further mocked in specific tests using return_value or side_effect
        self.generate_content_async = AsyncMock(return_value=MockGeminiResponse())


class TestAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Reset global gemini_model for isolation.
        # Tests that need a configured model should set this up via ensure_gemini_configured (mocked)
        # or by directly assigning a mock to agent.gemini_model.
        agent.gemini_model = None 
        
        self.mock_ctx = MagicMock(spec=Context)
        self.mock_ctx.log_info = AsyncMock()
        self.mock_ctx.log_error = AsyncMock()
        self.mock_ctx.log_warning = AsyncMock()


    @patch('tensordirectory.agent.os.getenv')
    @patch('tensordirectory.agent.genai.configure')
    @patch('tensordirectory.agent.genai.GenerativeModel') # Patch the class
    def test_01_ensure_gemini_configured_success(self, mock_gen_model_constructor, mock_genai_configure, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        # Ensure gemini_model is None before the call, so configuration proceeds.
        agent.gemini_model = None 
        
        # Have the constructor return our specific mock instance
        mock_model_instance = MockGenerativeModel()
        mock_gen_model_constructor.return_value = mock_model_instance

        ensure_gemini_configured() 
        
        mock_getenv.assert_called_with("GEMINI_API_KEY")
        mock_genai_configure.assert_called_with(api_key="fake_api_key")
        mock_gen_model_constructor.assert_called_with('gemini-pro') 
        self.assertIs(agent.gemini_model, mock_model_instance) # Check if the global is set

    @patch('tensordirectory.agent.os.getenv', return_value=None)
    def test_02_ensure_gemini_configured_no_api_key(self, mock_getenv):
        agent.gemini_model = None
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY not configured"):
            ensure_gemini_configured()
        mock_getenv.assert_called_with("GEMINI_API_KEY")
        self.assertIsNone(agent.gemini_model)


    # Tests for analyze_prompt_with_gemini
    @patch('tensordirectory.agent.ensure_gemini_configured', new_callable=AsyncMock) # Mock ensure_gemini_configured for these tests
    async def test_03_analyze_prompt_success_get_tensor(self, mock_ensure_config):
        # Setup a mock model and its response for this specific test
        mock_model_instance = MockGenerativeModel()
        gemini_json_response = json.dumps({"intent": "get_tensor", "entities": {"tensor_name": "t1"}})
        mock_model_instance.generate_content_async.return_value = MockGeminiResponse(text_content=gemini_json_response)
        agent.gemini_model = mock_model_instance # Directly assign the mock model

        result = await analyze_prompt_with_gemini("get tensor t1", self.mock_ctx)
        
        self.assertEqual(result, {"intent": "get_tensor", "entities": {"tensor_name": "t1"}})
        mock_model_instance.generate_content_async.assert_called_once()
        self.mock_ctx.log_info.assert_any_call(f"Raw Gemini response: {gemini_json_response}")
        self.mock_ctx.log_info.assert_any_call("Parsed Gemini response: {'intent': 'get_tensor', 'entities': {'tensor_name': 't1'}}")


    @patch('tensordirectory.agent.ensure_gemini_configured', new_callable=AsyncMock)
    async def test_04_analyze_prompt_gemini_returns_malformed_json(self, mock_ensure_config):
        mock_model_instance = MockGenerativeModel()
        malformed_json = "this is not json"
        mock_model_instance.generate_content_async.return_value = MockGeminiResponse(text_content=malformed_json)
        agent.gemini_model = mock_model_instance

        result = await analyze_prompt_with_gemini("some prompt", self.mock_ctx)
        
        self.assertIn("error", result)
        self.assertTrue(result["error"].startswith("Failed to parse Gemini response as JSON"))
        self.assertEqual(result["raw_response"], malformed_json)
        self.mock_ctx.log_error.assert_called()

    @patch('tensordirectory.agent.ensure_gemini_configured', new_callable=AsyncMock)
    async def test_05_analyze_prompt_gemini_api_error(self, mock_ensure_config):
        mock_model_instance = MockGenerativeModel()
        mock_model_instance.generate_content_async.side_effect = Exception("Gemini API Down")
        agent.gemini_model = mock_model_instance

        result = await analyze_prompt_with_gemini("any prompt", self.mock_ctx)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Gemini API call failed: Gemini API Down")
        self.mock_ctx.log_error.assert_called()
    
    @patch('tensordirectory.agent.ensure_gemini_configured', side_effect=ValueError("Config error for test"))
    async def test_06_analyze_prompt_gemini_not_configured_due_to_ensure_failure(self, mock_ensure_config_fails):
        agent.gemini_model = None 
        result = await analyze_prompt_with_gemini("any prompt", self.mock_ctx)
        self.assertIn("error", result)
        self.assertTrue(result["error"].startswith("Gemini model not configured: Config error for test"))
        self.mock_ctx.log_error.assert_called()


    # --- Tests for handle_query ---
    @patch('tensordirectory.agent.ensure_gemini_configured') 
    @patch('tensordirectory.agent.analyze_prompt_with_gemini')
    @patch('tensordirectory.storage.get_tensor_by_name')
    async def test_07_handle_query_get_tensor_found(self, mock_storage_get, mock_analyze, mock_ensure_config):
        mock_analyze.return_value = {"intent": "get_tensor", "entities": {"tensor_name": "t1"}}
        mock_tensor_data = np.array([1,2,3], dtype=np.int64) # Using a common default like int64
        mock_metadata = {"user_name": "t1", "description": "test desc", "uuid": "uuid-t1"}
        mock_storage_get.return_value = (mock_tensor_data, mock_metadata)

        response = await handle_query("get t1", None, self.mock_ctx)
        
        mock_analyze.assert_called_once_with("get t1", self.mock_ctx)
        mock_storage_get.assert_called_once_with("t1")
        self.assertEqual(response, "Tensor 't1' found. Metadata: {'user_name': 't1', 'description': 'test desc', 'uuid': 'uuid-t1'}. Shape: (3,), Dtype: int64.")

    @patch('tensordirectory.agent.ensure_gemini_configured')
    @patch('tensordirectory.agent.analyze_prompt_with_gemini')
    @patch('tensordirectory.storage.get_tensor_by_name')
    async def test_08_handle_query_get_tensor_not_found(self, mock_storage_get, mock_analyze, mock_ensure_config):
        mock_analyze.return_value = {"intent": "get_tensor", "entities": {"tensor_name": "t_not_exist"}}
        mock_storage_get.return_value = (None, None)
        
        response = await handle_query("get t_not_exist", None, self.mock_ctx)
        self.assertEqual(response, "Tensor 't_not_exist' not found.")

    @patch('tensordirectory.agent.ensure_gemini_configured')
    @patch('tensordirectory.agent.analyze_prompt_with_gemini')
    @patch('tensordirectory.storage.find_tensors')
    async def test_09_handle_query_find_tensors_found(self, mock_storage_find, mock_analyze, mock_ensure_config):
        mock_analyze.return_value = {"intent": "find_tensors_by_metadata", "entities": {"metadata_query": {"type":"feature"}}}
        mock_results = [
            (np.array([1]), {"user_name":"f1", "type":"feature"}),
            (np.array([2,3]), {"user_name":"f2", "type":"feature"})
        ]
        mock_storage_find.return_value = mock_results
        response = await handle_query("find features", None, self.mock_ctx)
        self.assertIn("Found 2 tensor(s) matching criteria. Examples: [Name='f1', Shape='(1,)', Dtype='int", response) # Dtype depends on np.array type

    @patch('tensordirectory.agent.ensure_gemini_configured')
    @patch('tensordirectory.agent.analyze_prompt_with_gemini')
    @patch('tensordirectory.storage.get_model_by_name')
    @patch('tensordirectory.storage.get_tensor_by_name')
    @patch('tensordirectory.storage.save_tensor')
    async def test_10_handle_query_run_inference_success(self, mock_save_tensor, mock_get_input_tensor, mock_get_model, mock_analyze, mock_ensure_config):
        mock_analyze.return_value = {
            "intent": "run_inference",
            "entities": {"model_name": "m1", "input_tensor_names": ["in1"], "output_tensor_name": "out1"}
        }
        model_code = "def predict(input_tensors_dict):\n  return input_tensors_dict['in1'] * 2" # Valid Python
        mock_get_model.return_value = {"code": model_code, "weights": None, "metadata": {"user_name": "m1"}}
        mock_input_data = np.array([10, 20])
        mock_get_input_tensor.return_value = (mock_input_data, {"user_name": "in1"})
        mock_save_tensor.return_value = "new-uuid-123" # Mock UUID for the saved output tensor

        response = await handle_query("run m1 with in1 output out1", None, self.mock_ctx)

        mock_get_model.assert_called_once_with("m1")
        mock_get_input_tensor.assert_called_once_with("in1")
        mock_save_tensor.assert_called_once()
        
        # Check the saved tensor data (args_save is a tuple, 0: name, 1: desc, 2: data)
        saved_call_args = mock_save_tensor.call_args[0] 
        self.assertEqual(saved_call_args[0], "out1")
        np.testing.assert_array_equal(saved_call_args[2], np.array([20, 40]))

        self.assertIn("Inference successful with model 'm1'. Output tensor saved as 'out1' (UUID: new-uuid-123).", response)
        self.mock_ctx.log_warning.assert_any_call("Attempting to execute model code for 'm1' using exec(). SECURITY RISK: Untrusted code execution.")


    @patch('tensordirectory.agent.ensure_gemini_configured')
    @patch('tensordirectory.agent.analyze_prompt_with_gemini')
    @patch('tensordirectory.storage.get_model_by_name')
    @patch('tensordirectory.storage.get_tensor_by_name') # Still need to mock this even if model exec fails earlier
    async def test_11_handle_query_run_inference_model_code_exec_exception(self, mock_get_input_tensor, mock_get_model, mock_analyze, mock_ensure_config):
        mock_analyze.return_value = {
            "intent": "run_inference", "entities": {"model_name": "m_err", "input_tensor_names": ["in1"]}
        }
        model_code_error = "def predict(input_tensors_dict):\n  raise ValueError('custom exec error')"
        mock_get_model.return_value = {"code": model_code_error, "metadata": {"user_name": "m_err"}}
        mock_get_input_tensor.return_value = (np.array([1]), {"user_name":"in1"}) # Input tensor is still fetched

        response = await handle_query("run m_err with in1", None, self.mock_ctx)
        
        self.assertEqual(response, "Error during model execution for 'm_err': custom exec error")
        self.mock_ctx.log_error.assert_any_call("Exception during model code execution for 'm_err': custom exec error", exc_info=True)

    @patch('tensordirectory.agent.ensure_gemini_configured')
    @patch('tensordirectory.agent.analyze_prompt_with_gemini')
    async def test_12_handle_query_gemini_analysis_error(self, mock_analyze, mock_ensure_config):
        mock_analyze.return_value = {"error": "Test Gemini Error", "raw_response": "raw"}
        response = await handle_query("any prompt", None, self.mock_ctx)
        self.assertEqual(response, "Error analyzing prompt: Test Gemini Error. Raw response: raw")

    @patch('tensordirectory.agent.ensure_gemini_configured', side_effect=ValueError("Config Test Fail"))
    async def test_13_handle_query_gemini_config_fails_in_handle_query(self, mock_ensure_config_fail):
        # This test ensures that if ensure_gemini_configured (called at the start of handle_query) fails,
        # it's caught and reported correctly.
        response = await handle_query("any prompt", None, self.mock_ctx)
        self.assertEqual(response, "Error: Could not configure AI model. Config Test Fail")
        self.mock_ctx.log_error.assert_called_once_with("Gemini configuration failed: Config Test Fail", exc_info=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
