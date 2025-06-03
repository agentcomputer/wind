# TensorDirectory API Playground

## Overview

The TensorDirectory API Playground provides a web-based interface to interactively explore and test the Model Context Protocol (MCP) API tools available in the TensorDirectory service. It allows users to select API tools, view their parameters, fill them out (with the help of demo data), execute the tools, and see the raw MCP request and response.

This is particularly useful for developers integrating with the API, data scientists exploring available models and tensors, and anyone wanting to understand the capabilities of the TensorDirectory service without writing client code.

## Running the Playground

The API Playground is built using FastAPI and runs as a web server.

1.  **Ensure Dependencies are Installed:**
    Make sure all dependencies, including those for the main TensorDirectory service and the playground (`fastapi`, `uvicorn`), are installed:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Playground Server:**
    You can run the playground server directly using its main script:
    ```bash
    python -m tensordirectory.playground.main
    ```
    This will typically start the server on `http://localhost:8080`. The server is configured to reload on code changes.

3.  **Access in Browser:**
    Open your web browser and navigate to [http://localhost:8080](http://localhost:8080).

**Note:** For the playground to fully execute MCP tools that interact with storage (like `upload_tensor`, `list_tensors`, etc.) or the AI agent (`query_tensor_directory`), the main TensorDirectory components (HDF5 storage initialization, Gemini AI configuration if used by `query_tensor_directory`) must be functional. The playground's FastAPI backend directly calls the MCP tools defined in `tensordirectory.mcp_interface`. The HDF5 file used will be `tensor_directory.hdf5` in the project root, unless configured otherwise for the main storage module.

## Features

*   **Tool Discovery:** Automatically lists all available MCP API tools.
*   **Dynamic Parameter Forms:** Generates input forms tailored to the selected tool's parameters based on their Pydantic models or defined schemas.
*   **Demo Data:** Provides pre-filled example data for common tools to simplify testing and demonstrate expected input formats.
*   **Request Visualization:** Shows the constructed JSON payload that will be sent to the MCP tool.
*   **Response Display:** Displays the JSON response (or error) received from the MCP tool execution.
*   **User-Friendly Interface:** Designed for ease of use with a clean and responsive layout.

## Using the Playground

1.  **Select an API Tool:** Choose the MCP tool you want to interact with from the "Select API Tool" dropdown menu.
2.  **Fill Parameters:** The form for the tool's parameters will appear below the dropdown. Fill these in manually based on the requirements. Required fields are marked with an asterisk (*).
3.  **Use Demo Data (Optional):** If demo data is available for the selected tool, options to load it will appear in the "Demo Data" section. Clicking a demo data button will populate the form fields with the example data. This is a good way to see the expected structure for complex inputs like JSON.
4.  **Execute:** Click the "Execute Tool" button.
5.  **View Request & Response:** The generated MCP request (what the playground sends to its backend) and the response from the tool execution (or any error messages) will be displayed in the "MCP Request" and "MCP Response" boxes on the right.

## Troubleshooting

*   **"Tool execution failed" / 500 errors / 422 Validation Errors:**
    *   **Storage Issues:** Ensure the HDF5 file (`tensor_directory.hdf5` by default) is initialized and writable if the tool interacts with storage (e.g., `upload_tensor`, `delete_model`). If the file doesn't exist, `storage._initialize_hdf5()` (called by storage functions) should create it, but permissions issues might prevent this.
    *   **AI Agent Issues (for `query_tensor_directory`):** Ensure your `GEMINI_API_KEY` is correctly set up in your environment or `.env` file if the query involves the AI agent.
    *   **Invalid Arguments (422 Error):** If you get a "Validation Error", the "MCP Response" box will show details about which fields are incorrect. Check the expected types (e.g., numbers for numeric fields, valid JSON for fields expecting JSON arrays/objects).
    *   **Server Console:** Check the console output of the `python -m tensordirectory.playground.main` server for more detailed error messages from the FastAPI backend or the MCP tools themselves.
*   **Playground UI not loading correctly or Demo Data/Forms not appearing:**
    *   Ensure the `python -m tensordirectory.playground.main` command is running without errors in your terminal.
    *   Check your browser's developer console (usually F12) for any JavaScript errors that might prevent the UI from rendering or functioning correctly.
    *   Ensure you have the latest code and have installed all dependencies from `requirements.txt`.
*   **"Tool not found" / 404 errors for `/api/execute_tool` or `/api/tools/{tool_name}/schema`:**
    *   This usually indicates an issue with the tool name being passed from the frontend to the backend or an unexpected problem with the `mcp_server` instance. Verify the tool name is correct and that `mcp_server` is correctly initialized.
