# TensorDirectory Service: Sample Data & Interaction Examples

This document provides sample tensor data, model code, and conceptual examples of how to interact with the TensorDirectory service using an MCP (Model Context Protocol) client.

## 1. Sample Tensor Data

Here are a few examples of tensor data that you can upload to the TensorDirectory. The `tensor_data` is represented as Python lists.

### a. 1D Vector: `sample_vector_alpha`

*   **Name:** `"sample_vector_alpha"`
*   **Description:** `"A 1D sample vector for demonstration."`
*   **Data (Python list representation):**
    ```python
    [1.0, 2.5, -3.3, 4.1, 0.0]
    ```

### b. 2D Matrix: `sample_matrix_beta`

*   **Name:** `"sample_matrix_beta"`
*   **Description:** `"A 2D sample matrix."`
*   **Data (Python list of lists representation):**
    ```python
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ```

### c. Tiny RGB Image Tensor: `tiny_rgb_image`

*   **Name:** `"tiny_rgb_image"`
*   **Description:** `"A 1x2x3 RGB tensor representing a tiny image (1 pixel high, 2 pixels wide, 3 color channels)."`
*   **Data (Python list of lists of lists representation):**
    ```python
    [[[255, 0, 0], [0, 255, 0]]]  # Represents a red pixel and a green pixel
    ```

## 2. Sample Model Code

Models can be uploaded as Python code strings. The code **must** define a function `predict(input_tensors_dict: dict[str, np.ndarray]) -> np.ndarray`.

### a. Simple Scaler Model: `simple_scaler`

*   **Name:** `"simple_scaler"`
*   **Description:** `"A model that multiplies the input tensor by a scalar value (e.g., 2.0)."`
*   **Model Code (string):**
    ```python
    # Model Code for 'simple_scaler'
    import numpy as np
    def predict(input_tensors_dict: dict[str, np.ndarray]) -> np.ndarray:
        # Assumes one input tensor is provided in the dict.
        # For a more robust model, you might expect specific key names.
        if not input_tensors_dict:
            raise ValueError("Input tensor dictionary is empty for simple_scaler.")

        input_tensor_key = list(input_tensors_dict.keys())[0] # Get the first key
        input_tensor = input_tensors_dict[input_tensor_key]

        scale_factor = 2.0 # Example scale factor
        return input_tensor * scale_factor
    ```

### b. Tensor Adder Model: `tensor_adder`

*   **Name:** `"tensor_adder"`
*   **Description:** `"A model that adds two input tensors, expecting keys 'input_A' and 'input_B'."`
*   **Model Code (string):**
    ```python
    # Model Code for 'tensor_adder'
    import numpy as np
    def predict(input_tensors_dict: dict[str, np.ndarray]) -> np.ndarray:
        if 'input_A' not in input_tensors_dict or 'input_B' not in input_tensors_dict:
            raise ValueError("Missing 'input_A' or 'input_B' in input_tensors_dict for tensor_adder model.")

        tensor_a = input_tensors_dict['input_A']
        tensor_b = input_tensors_dict['input_B']

        if tensor_a.shape != tensor_b.shape:
            raise ValueError(f"Shapes of input_A {tensor_a.shape} and input_B {tensor_b.shape} must match for addition.")

        return tensor_a + tensor_b
    ```

## 3. Conceptual Python MCP Client Interaction Examples

The following snippets illustrate how a hypothetical Python MCP client might interact with the TensorDirectory service's tools. The exact client syntax may vary.

*(Assume `mcp_client` is an initialized and connected MCP client instance)*

### a. Uploading a Sample Tensor

```python
# Uploading 'sample_vector_alpha'
tensor_to_upload = {
    "name": "sample_vector_alpha",
    "description": "A 1D sample vector for demonstration.",
    "tensor_data": [1.0, 2.5, -3.3, 4.1, 0.0]
}

try:
    response = await mcp_client.call_tool(
        tool_name="upload_tensor",
        args=tensor_to_upload
    )
    print(f"Upload Tensor Response: {response}")
    # Expected: {'uuid': 'some-uuid', 'name': 'sample_vector_alpha', 'message': 'Tensor uploaded successfully'}
    uploaded_vector_uuid = response.get('uuid')
except Exception as e:
    print(f"Error uploading tensor: {e}")

# You can similarly upload 'sample_matrix_beta' and 'tiny_rgb_image'
# For example, to upload sample_matrix_beta for the adder model:
tensor_matrix_A_args = {
    "name": "matrix_A_for_adder",
    "description": "First input for tensor_adder model.",
    "tensor_data": [[1, 2], [3, 4]]
}
# response_A = await mcp_client.call_tool(tool_name="upload_tensor", args=tensor_matrix_A_args)
# print(f"Uploaded matrix_A_for_adder: {response_A}")

tensor_matrix_B_args = {
    "name": "matrix_B_for_adder",
    "description": "Second input for tensor_adder model.",
    "tensor_data": [[5, 6], [7, 8]]
}
# response_B = await mcp_client.call_tool(tool_name="upload_tensor", args=tensor_matrix_B_args)
# print(f"Uploaded matrix_B_for_adder: {response_B}")
```

### b. Uploading a Sample Model

```python
# Uploading 'simple_scaler' model
scaler_model_to_upload = {
    "name": "simple_scaler",
    "description": "A model that multiplies the input tensor by 2.0.",
    "model_code": "import numpy as np\ndef predict(input_tensors_dict):\n  input_tensor_key = list(input_tensors_dict.keys())[0]\n  input_tensor = input_tensors_dict[input_tensor_key]\n  return input_tensor * 2.0"
}

try:
    response = await mcp_client.call_tool(
        tool_name="upload_model",
        args=scaler_model_to_upload
    )
    print(f"Upload Model Response: {response}")
    # Expected: {'uuid': 'some-model-uuid', 'name': 'simple_scaler', 'message': 'Model uploaded successfully'}
    scaler_model_uuid = response.get('uuid')
except Exception as e:
    print(f"Error uploading model: {e}")

# Similarly, upload 'tensor_adder' model
adder_model_to_upload = {
    "name": "tensor_adder",
    "description": "A model that adds two input tensors ('input_A' and 'input_B').",
    "model_code": "import numpy as np\ndef predict(input_tensors_dict):\n  if 'input_A' not in input_tensors_dict or 'input_B' not in input_tensors_dict:\n    raise ValueError(\"Missing 'input_A' or 'input_B'\")\n  tensor_a = input_tensors_dict['input_A']\n  tensor_b = input_tensors_dict['input_B']\n  if tensor_a.shape != tensor_b.shape:\n    raise ValueError(\"Shapes must match\")\n  return tensor_a + tensor_b"
}
# try:
#    response_adder = await mcp_client.call_tool(tool_name="upload_model", args=adder_model_to_upload)
#    print(f"Uploaded tensor_adder model: {response_adder}")
# except Exception as e:
#    print(f"Error uploading tensor_adder model: {e}")
```

### c. Querying for a Tensor

```python
# Assuming 'sample_vector_alpha' was uploaded
query_prompt = "Get tensor named 'sample_vector_alpha'"
query_args = {"prompt": query_prompt}

try:
    response_str = await mcp_client.call_tool(
        tool_name="query_tensor_directory",
        args=query_args
    )
    print(f"Query Response: {response_str}")
    # Expected: "Tensor 'sample_vector_alpha' found. Metadata: {...}. Shape: (5,), Dtype: float64."
except Exception as e:
    print(f"Error querying directory: {e}")
```

### d. Running Inference with a Sample Model and Tensor

```python
# Assuming 'sample_vector_alpha' and 'simple_scaler' model were uploaded.
# The agent needs to identify 'sample_vector_alpha' as the input for 'simple_scaler'.
inference_prompt_scaler = "Run the 'simple_scaler' model using 'sample_vector_alpha' as input, and name the output 'scaled_vector_result'."

inference_args_scaler = {"prompt": inference_prompt_scaler}

try:
    response_str = await mcp_client.call_tool(
        tool_name="query_tensor_directory",
        args=inference_args_scaler
    )
    print(f"Inference Response (Scaler): {response_str}")
    # Expected: "Inference successful with model 'simple_scaler'. Output tensor saved as 'scaled_vector_result' (UUID: ...)."
except Exception as e:
    print(f"Error running inference with scaler: {e}")

# Example for the 'tensor_adder' model:
# First, ensure 'matrix_A_for_adder' and 'matrix_B_for_adder' (from section 3a examples) are uploaded.
# Then, the prompt would be something like:
inference_prompt_adder = "Use model 'tensor_adder' with 'matrix_A_for_adder' as input_A and 'matrix_B_for_adder' as input_B. Call the output 'summed_matrix'."
inference_args_adder = {"prompt": inference_prompt_adder}

# try:
#    response_str_adder = await mcp_client.call_tool(
#        tool_name="query_tensor_directory",
#        args=inference_args_adder
#    )
#    print(f"Inference Response (Adder): {response_str_adder}")
#    # Expected: "Inference successful with model 'tensor_adder'. Output tensor saved as 'summed_matrix' (UUID: ...)."
# except Exception as e:
#    print(f"Error running inference with adder: {e}")

```
