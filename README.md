# TensorDirectory Service

## Overview

TensorDirectory is a service designed to store, manage, and retrieve tensor data and inference models. It provides an MCP (Model Context Protocol) interface for programmatic interaction and leverages an AI agent (powered by Google Gemini) to understand natural language queries for accessing and manipulating the stored data. Tensors and model weights are stored efficiently in HDF5 files.

## Features

*   **MCP Interface:** Exposes resources and tools for interacting with the directory via the Model Context Protocol.
*   **HDF5 Storage:** Uses HDF5 for persistent storage of large tensor data and model components.
*   **AI-Powered Queries:** An integrated AI agent (using Google Gemini) interprets natural language prompts to perform actions like retrieving tensors, finding models, or running inference.
*   **Tensor Upload & Management:** Allows uploading tensors with associated metadata.
*   **Model Upload & Management:** Supports uploading inference models, including their Python code and/or weights.
*   **Basic Inference Execution:** Capable of executing models defined by Python code (with important security considerations).

## Project Structure

*   `tensordirectory/`: Contains the core source code for the service.
    *   `main.py`: Entry point to run the MCP server.
    *   `mcp_interface.py`: Defines the MCP resources and tools.
    *   `agent.py`: Implements the AI agent logic using Gemini for query processing and model execution.
    *   `storage.py`: Handles the HDF5 file operations for storing and retrieving tensors and models.
    *   `__init__.py`: Makes `tensordirectory` a Python package.
*   `tests/`: Contains unit tests for the service components.
    *   `test_storage.py`: Tests for HDF5 storage operations.
    *   `test_mcp_interface.py`: Tests for the MCP request handlers.
    *   `test_agent.py`: Tests for the AI agent logic.
    *   `__init__.py`: Makes `tests` a Python package.
*   `README.md`: This file - provides an overview and setup instructions.
*   `API.md`: Detailed documentation of the MCP API endpoints.
*   `requirements.txt`: Lists Python dependencies.
*   `.env` (example): File for storing environment variables like `GEMINI_API_KEY`.

## Setup Instructions

### Prerequisites

*   Python 3.9+

### 1. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 2. Install Dependencies

Create a `requirements.txt` file (if not already present) with the following content:

```txt
numpy
h5py
google-generativeai
python-dotenv
modelcontextprotocol
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

The AI agent requires access to the Google Gemini API. You need to set up an API key:

1.  Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Create a file named `.env` in the project root directory (alongside `README.md`).
3.  Add your API key to the `.env` file like this:

    ```env
    GEMINI_API_KEY=your_actual_gemini_api_key_here
    ```

    The application uses `python-dotenv` to load this key automatically.

## Running the Server

To start the TensorDirectory MCP server, run the following command from the **project root directory**:

```bash
python -m tensordirectory.main
```

You should see log messages indicating that the HDF5 storage is initialized, the Gemini model is configured, and the MCP server is starting. Clients can then connect to the service using an MCP client.

## Running Tests

To run the unit tests, execute the following command from the **project root directory**:

```bash
python -m unittest discover tests
```
For more detailed output:
```bash
python -m unittest discover -v tests
```

## Security Notes

**CRITICAL SECURITY WARNING:**

The current implementation of model execution in `agent.py` uses Python's `exec()` function to run model code provided by users. **Using `exec()` with untrusted code is a significant security risk and can lead to arbitrary code execution on the server.**

**This feature, as implemented, is NOT SUITABLE FOR PRODUCTION ENVIRONMENTS where models might be uploaded from untrusted sources.** For production use, model execution should be handled via sandboxing, secure execution environments (like containers with restricted permissions), or by using models in standardized, safe formats (e.g., ONNX, TensorFlow Lite) loaded via trusted libraries.

## Basic Usage / MCP API Overview

TensorDirectory exposes its functionality through the Model Context Protocol (MCP). Key operations include:

*   **Uploading Tensors:** Store tensor data along with metadata.
*   **Uploading Models:** Store inference models (Python code and/or weights).
*   **Querying:** Use natural language to find tensors, models, or execute inference tasks.

For detailed information on the available MCP resources, tools, their parameters, and example requests/responses, please refer to the [API.md](API.md) documentation.

---

## API Playground

This project includes an interactive API Playground to easily test and explore the MCP API tools. For details on how to run and use it, please see the [API Playground Documentation](./PLAYGROUND.md).
