# TensorDirectory MCP API Documentation

## Introduction

This document provides details on the Model Context Protocol (MCP) interface for the TensorDirectory service. This interface allows programmatic interaction for uploading and managing tensors and models, as well as querying the directory using an AI agent.

---

## Tool: `upload_tensor`

**Description:**
This tool is used to upload tensor data along with its metadata to the TensorDirectory.

**Method:**
MCP Tool. Clients typically invoke this by providing the tool name (`upload_tensor`) and an arguments object.

**Parameters:**
The tool accepts a single argument object (e.g., `args` when calling the tool function) with the following fields, corresponding to the `TensorUploadArgs` Pydantic model:

*   `name: str` (Required)
    *   A user-defined name for the tensor. This name can be used for later retrieval.
*   `description: str` (Required)
    *   A textual description of the tensor, its source, or its purpose.
*   `tensor_data: list` (Required)
    *   The actual tensor data, represented as a nested list of numbers. This list should be structured in a way that can be directly converted into a NumPy array (e.g., a list of lists for a 2D tensor).

**Example Arguments for Tool Call (conceptual JSON representation of `TensorUploadArgs`):**

```json
{
    "name": "my_feature_vector_v2",
    "description": "Feature vector extracted from image 'img_1024.jpg' using ResNet50",
    "tensor_data": [
        [0.15, 0.25, 0.35, -0.45],
        [0.45, 0.55, -0.65, 0.75],
        [0.85, -0.95, 1.05, 0.05]
    ]
}
```

**Example Success Response (JSON body returned by the tool function):**

Upon successful upload, the tool returns a JSON object containing the generated unique ID (UUID) for the tensor, its given name, and a success message.

```json
{
    "uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "name": "my_feature_vector_v2",
    "message": "Tensor uploaded successfully"
}
```

**Example Error Response (JSON body returned by the tool function):**

If an error occurs (e.g., invalid data format, storage failure), an error message is returned. Pydantic validation errors for the arguments object are typically handled by the FastMCP framework before the tool function is called, resulting in a framework-level error response (e.g., HTTP 422).

```json
{
    "error": "tensor_data for 'my_feature_vector_v2' must be a non-empty list."
}
```
```json
{
    "error": "An unexpected error occurred while uploading tensor 'my_feature_vector_v2': [Details of storage error]"
}
```

---

## Tool: `upload_model`

**Description:**
This tool is used to upload inference models to the TensorDirectory. A model can consist of Python code, NumPy array weights, or both.

**Method:**
MCP Tool. Clients typically invoke this by providing the tool name (`upload_model`) and an arguments object.

**Parameters:**
The tool accepts a single argument object (e.g., `args` when calling the tool function) with the following fields, corresponding to the `ModelUploadArgs` Pydantic model:

*   `name: str` (Required)
    *   A user-defined name for the model.
*   `description: str` (Required)
    *   A textual description of the model, its architecture, or its intended use.
*   `model_weights: Optional[list] = None` (Optional)
    *   The model's weights, represented as a nested list of numbers (convertible to a NumPy array). Defaults to `None`.
*   `model_code: Optional[str] = None` (Optional)
    *   A string containing the Python code for the model.
    *   **Important:** If `model_code` is provided, it **must** define a Python function with the signature `predict(input_tensors_dict: dict[str, np.ndarray]) -> np.ndarray`. The `input_tensors_dict` will be a dictionary where keys are user-specified input tensor names (as identified by the AI agent or during an inference call) and values are the corresponding NumPy arrays. The function must return a single NumPy array as the inference result.
    *   Defaults to `None`. At least one of `model_weights` or `model_code` must be provided.

**Example Arguments for Tool Call (conceptual JSON representation of `ModelUploadArgs` for a code-only model):**

```json
{
    "name": "image_enhancer_v1",
    "description": "Simple model to adjust image brightness, defined in Python.",
    "model_code": "import numpy as np\\ndef predict(input_tensors_dict):\\n  image = input_tensors_dict['input_image']\\n  return np.clip(image * 1.2 + 10, 0, 255)"
}
```

**Example Arguments for Tool Call (conceptual JSON for a weights-only model):**

```json
{
    "name": "embedding_lookup_table",
    "description": "Pre-trained embedding weights.",
    "model_weights": [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]
}
```

**Example Success Response (JSON body returned by the tool function):**

```json
{
    "uuid": "f0e1d2c3-b4a5-6789-0123-456789abcdef",
    "name": "image_enhancer_v1",
    "message": "Model uploaded successfully"
}
```

**Example Error Response (JSON body returned by the tool function):**

```json
{
    "error": "Either model_weights or model_code must be provided."
}
```
```json
{
    "error": "An unexpected error occurred while uploading model 'image_enhancer_v1': [Details of storage error]"
}
```

---

## Tool: `query_tensor_directory`

**Description:**
This tool allows users to interact with the TensorDirectory using natural language prompts. The AI agent backend interprets these prompts to perform actions such as retrieving tensor/model information or executing model inference.

**Parameters:**

*   `prompt: str` (Required)
    *   The natural language query. Examples:
        *   "Find tensor named 'my_feature_vector_v2'"
        *   "What is the shape of tensor 'my_feature_vector_v2'?"
        *   "Search for tensors with dtype float32"
        *   "Load model 'image_enhancer_v1'"
        *   "Run model 'image_enhancer_v1' on tensor 'input_image_data' and call the output 'enhanced_output'"
*   `params: dict | None` (Optional)
    *   Additional parameters for the query, which may be used by the agent for more complex operations. Currently, this is not heavily used but is available for future extensions.

**Example Request (Conceptual Client Call):**

```python
# Example using a hypothetical MCP client library
# response_string = await client.call_tool(
#     tool_name="query_tensor_directory",
#     args={"prompt": "What are the details for tensor 'my_feature_vector_v2'?", "params": {}}
# )
```

**Example Responses (String):**

The tool returns a string which can be a direct answer, a summary, a status message, or an error message.

*   **Tensor Retrieval:**
    `"Tensor 'my_feature_vector_v2' found. Metadata: {'user_name': 'my_feature_vector_v2', ...}. Shape: (3, 4), Dtype: float64."`
    `"Tensor 'unknown_tensor' not found."`

*   **Tensor Search:**
    `"Found 2 tensor(s) matching criteria. Examples: [Name='t1', Shape='(10,)', Dtype='float32'; Name='t2', Shape='(20, 5)', Dtype='float32']"`
    `"No tensors found matching your criteria."`

*   **Model Inference:**
    `"Inference successful with model 'image_enhancer_v1'. Output tensor saved as 'enhanced_output_20231027_143000_abcdef12' (UUID: ...)."`
    `"Model 'image_enhancer_v1' found, but it only has weights. Direct execution of weights-only models is not yet supported by this agent."`
    `"Error: Input tensor 'input_image_data' not found for inference."`
    `"Error during model execution for 'image_enhancer_v1': [Details of execution error]"`

*   **General Errors / Unknown Intent:**
    `"Error analyzing prompt: Gemini model not configured."`
    `"The intent of your request is unclear or not supported. Please try rephrasing. User prompt: 'tell me a joke'"`

---
**Security Note on `model_code` Execution:**
As highlighted in the main `README.md`, the execution of user-provided `model_code` via `exec()` in the agent is a **significant security risk**. This functionality should not be used in production environments with untrusted model code.

---

## Tool: `list_tensors`

**Description:**
This tool lists tensors stored in the TensorDirectory, allowing for optional filtering by name and pagination.

**Method:**
MCP Tool. Clients invoke this by providing the tool name (`list_tensors`) and an arguments object.

**Parameters:**
The tool accepts a single argument object (e.g., `args`) with the following fields, corresponding to the `ListArgs` Pydantic model:

*   `filter_by_name_contains: Optional[str] = None` (Optional)
    *   A string to filter tensors whose names contain this substring. Case-sensitive.
*   `limit: int = Field(default=100, gt=0, le=1000)` (Optional)
    *   The maximum number of tensor metadata entries to return. Defaults to 100. Must be between 1 and 1000.
*   `offset: int = Field(default=0, ge=0)` (Optional)
    *   The number of tensor metadata entries to skip before starting to collect the result set. Defaults to 0. Must be greater than or equal to 0.

**Example Arguments for Tool Call (conceptual JSON representation of `ListArgs`):**

```json
{
    "filter_by_name_contains": "feature_vector",
    "limit": 50,
    "offset": 0
}
```

**Example Success Response (JSON body returned by the tool function, corresponding to `ListTensorsResponse`):**

```json
{
    "tensors": [
        {
            "uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
            "user_name": "my_feature_vector_v2",
            "description": "Feature vector extracted from image 'img_1024.jpg' using ResNet50",
            "creation_date": "2023-10-28T10:30:00.123456",
            "original_dtype": "float64",
            "original_shape": "(3, 4)"
        },
        {
            "uuid": "b2c3d4e5-f6a7-8901-2345-678901bcdef0",
            "user_name": "another_feature_vector",
            "description": "Some other features",
            "creation_date": "2023-10-29T11:00:00.567890",
            "original_dtype": "int32",
            "original_shape": "(100,)"
        }
    ],
    "total_items_in_collection": 2,
    "offset": 0,
    "limit": 50
}
```

**Example Error Response (JSON body returned by the tool function if an internal error occurs):**
(Note: Pydantic validation errors for arguments are typically handled by FastMCP at a higher level.)

```json
{
    "tensors": [],
    "total_items_in_collection": 0,
    "offset": 0,
    "limit": 50
}
```

---

## Tool: `list_models`

**Description:**
This tool lists models stored in the TensorDirectory, allowing for optional filtering by name and pagination.

**Method:**
MCP Tool. Clients invoke this by providing the tool name (`list_models`) and an arguments object.

**Parameters:**
The tool accepts a single argument object (e.g., `args`) with the following fields, corresponding to the `ListArgs` Pydantic model:

*   `filter_by_name_contains: Optional[str] = None` (Optional)
    *   A string to filter models whose names contain this substring. Case-sensitive.
*   `limit: int = Field(default=100, gt=0, le=1000)` (Optional)
    *   The maximum number of model metadata entries to return. Defaults to 100. Must be between 1 and 1000.
*   `offset: int = Field(default=0, ge=0)` (Optional)
    *   The number of model metadata entries to skip before starting to collect the result set. Defaults to 0. Must be greater than or equal to 0.

**Example Arguments for Tool Call (conceptual JSON representation of `ListArgs`):**

```json
{
    "filter_by_name_contains": "enhancer",
    "limit": 10,
    "offset": 0
}
```

**Example Success Response (JSON body returned by the tool function, corresponding to `ListModelsResponse`):**

```json
{
    "models": [
        {
            "uuid": "f0e1d2c3-b4a5-6789-0123-456789abcdef",
            "user_name": "image_enhancer_v1",
            "description": "Simple model to adjust image brightness, defined in Python.",
            "upload_date": "2023-10-28T12:00:00.123456",
            "has_code": true,
            "has_weights": false
        },
        {
            "uuid": "g1h2i3j4-k5l6-m7n8-o9p0-qrstuvwxyz01",
            "user_name": "super_resolution_enhancer",
            "description": "Advanced SRGAN model.",
            "upload_date": "2023-10-29T15:00:00.567890",
            "has_code": false,
            "has_weights": true
        }
    ],
    "total_items_in_collection": 2,
    "offset": 0,
    "limit": 10
}
```

**Example Error Response (JSON body returned by the tool function if an internal error occurs):**

```json
{
    "models": [],
    "total_items_in_collection": 0,
    "offset": 0,
    "limit": 10
}
```

---

## Tool: `delete_tensor`

**Description:**
This tool deletes a tensor from the TensorDirectory by its user-defined name or its system-generated UUID.

**Method:**
MCP Tool. Clients invoke this by providing the tool name (`delete_tensor`) and an arguments object.

**Parameters:**
The tool accepts a single argument object (e.g., `args`) with the following fields, corresponding to the `DeleteArgs` Pydantic model:

*   `name_or_uuid: str` (Required)
    *   The name or UUID of the tensor to be deleted.

**Example Arguments for Tool Call (conceptual JSON representation of `DeleteArgs`):**

Using name:
```json
{
    "name_or_uuid": "my_feature_vector_v2"
}
```

Using UUID:
```json
{
    "name_or_uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
}
```

**Example Success Response (JSON body returned by the tool function):**

```json
{
    "success": true,
    "message": "Tensor 'my_feature_vector_v2' (UUID: a1b2c3d4-e5f6-7890-1234-567890abcdef) deleted successfully."
}
```

**Example Error Response (JSON body returned by the tool function):**

Tensor not found by name:
```json
{
    "success": false,
    "message": "Tensor 'non_existent_tensor' not found by name."
}
```

Tensor not found by UUID (or delete failed for other reasons):
```json
{
    "success": false,
    "message": "Tensor UUID 'a1b2c3d4-e5f6-7890-1234-000000000000' not found or delete failed."
}
```

Storage file not found (system-level issue):
```json
{
    "success": false,
    "message": "Storage file not found."
}
```

---

## Tool: `delete_model`

**Description:**
This tool deletes a model from the TensorDirectory by its user-defined name or its system-generated UUID.

**Method:**
MCP Tool. Clients invoke this by providing the tool name (`delete_model`) and an arguments object.

**Parameters:**
The tool accepts a single argument object (e.g., `args`) with the following fields, corresponding to the `DeleteArgs` Pydantic model:

*   `name_or_uuid: str` (Required)
    *   The name or UUID of the model to be deleted.

**Example Arguments for Tool Call (conceptual JSON representation of `DeleteArgs`):**

Using name:
```json
{
    "name_or_uuid": "image_enhancer_v1"
}
```

Using UUID:
```json
{
    "name_or_uuid": "f0e1d2c3-b4a5-6789-0123-456789abcdef"
}
```

**Example Success Response (JSON body returned by the tool function):**

```json
{
    "success": true,
    "message": "Model 'image_enhancer_v1' (UUID: f0e1d2c3-b4a5-6789-0123-456789abcdef) deleted successfully."
}
```

**Example Error Response (JSON body returned by the tool function):**

Model not found by name:
```json
{
    "success": false,
    "message": "Model 'unknown_model' not found by name."
}
```

Model not found by UUID:
```json
{
    "success": false,
    "message": "Model UUID 'f0e1d2c3-b4a5-6789-0123-000000000000' not found or delete failed."
}
```

---

## Tool: `update_tensor_metadata`

**Description:**
This tool updates the metadata of an existing tensor in the TensorDirectory, identified by its name or UUID.

**Method:**
MCP Tool. Clients invoke this by providing the tool name (`update_tensor_metadata`) and an arguments object.

**Parameters:**
The tool accepts a single argument object (e.g., `args`) with the following fields, corresponding to the `UpdateMetadataArgs` Pydantic model:

*   `name_or_uuid: str` (Required)
    *   The name or UUID of the tensor whose metadata is to be updated.
*   `metadata_updates: Dict[str, Any]` (Required)
    *   A dictionary containing the metadata fields to update and their new values.
    *   Currently, updatable fields for tensors typically include `user_name` and `description`. Other fields like `uuid`, `creation_date`, `original_dtype`, `original_shape` are generally considered immutable or system-managed post-creation. The storage layer will determine which fields can be updated.

**Example Arguments for Tool Call (conceptual JSON representation of `UpdateMetadataArgs`):**

```json
{
    "name_or_uuid": "my_feature_vector_v2",
    "metadata_updates": {
        "description": "Updated description: Feature vector from ResNet50, normalized.",
        "user_name": "my_feature_vector_v2_normalized"
    }
}
```

**Example Success Response (JSON body returned by the tool function):**

The response includes the full, updated metadata for the tensor.

```json
{
    "success": true,
    "metadata": {
        "uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "user_name": "my_feature_vector_v2_normalized",
        "description": "Updated description: Feature vector from ResNet50, normalized.",
        "creation_date": "2023-10-28T10:30:00.123456",
        "original_dtype": "float64",
        "original_shape": "(3, 4)"
    }
}
```

**Example Error Response (JSON body returned by the tool function):**

Tensor not found:
```json
{
    "success": false,
    "message": "Tensor 'non_existent_tensor' not found by name for update."
}
```

Update failed (e.g., UUID not found in storage, or internal error during update):
```json
{
    "success": false,
    "message": "Tensor UUID 'a1b2c3d4-e5f6-7890-1234-000000000000' not found or update failed."
}
```

---

## Tool: `update_model_metadata`

**Description:**
This tool updates the metadata of an existing model in the TensorDirectory, identified by its name or UUID.

**Method:**
MCP Tool. Clients invoke this by providing the tool name (`update_model_metadata`) and an arguments object.

**Parameters:**
The tool accepts a single argument object (e.g., `args`) with the following fields, corresponding to the `UpdateMetadataArgs` Pydantic model:

*   `name_or_uuid: str` (Required)
    *   The name or UUID of the model whose metadata is to be updated.
*   `metadata_updates: Dict[str, Any]` (Required)
    *   A dictionary containing the metadata fields to update and their new values.
    *   Currently, updatable fields for models typically include `user_name` and `description`. Fields like `uuid`, `upload_date`, `has_code`, `has_weights` are generally immutable or reflect the stored content. The storage layer determines which fields are updatable.

**Example Arguments for Tool Call (conceptual JSON representation of `UpdateMetadataArgs`):**

```json
{
    "name_or_uuid": "image_enhancer_v1",
    "metadata_updates": {
        "description": "Simple model to adjust image brightness and contrast. Python based.",
        "user_name": "image_brightness_contrast_v1.1"
    }
}
```

**Example Success Response (JSON body returned by the tool function):**

The response includes the full, updated metadata for the model.

```json
{
    "success": true,
    "metadata": {
        "uuid": "f0e1d2c3-b4a5-6789-0123-456789abcdef",
        "user_name": "image_brightness_contrast_v1.1",
        "description": "Simple model to adjust image brightness and contrast. Python based.",
        "upload_date": "2023-10-28T12:00:00.123456",
        "has_code": true,
        "has_weights": false
    }
}
```

**Example Error Response (JSON body returned by the tool function):**

Model not found:
```json
{
    "success": false,
    "message": "Model 'unknown_model_v2' not found by name for update."
}
```

Update failed:
```json
{
    "success": false,
    "message": "Model UUID 'f0e1d2c3-b4a5-6789-0123-000000000000' not found or update failed."
}
```
