"""
Main entry point for the TensorDirectory service.

This script initializes necessary components like logging, HDF5 storage,
and the Gemini AI model configuration. It then starts the MCP (Model Context Protocol)
server, allowing clients to interact with the TensorDirectory service.

To run the server, execute this script from the project root:
    python -m tensordirectory.main
"""
# main.py

import logging
import sys

# Attempt to set up project imports.
# This assumes the script is run in an environment where 'tensordirectory' package is accessible.
# For example, running `python -m tensordirectory.main` from the project root,
# or having the project root in PYTHONPATH.
try:
    from tensordirectory.mcp_interface import mcp_server
    from tensordirectory.agent import ensure_gemini_configured
    from tensordirectory.storage import _initialize_hdf5, HDF5_FILE_NAME
except ImportError as e:
    # Provide a helpful message if imports fail, common in complex project structures.
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
    logging.error(f"Failed to import TensorDirectory modules: {e}")
    logging.error("Please ensure the project is installed correctly or PYTHONPATH is set up.")
    logging.error("Try running as a module: `python -m tensordirectory.main` from the project root.")
    sys.exit(1)

# Configure basic logging
# MCP server itself will also have logging, this is for main.py specific messages.
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting TensorDirectory Service...")

    # 1. Initialize HDF5 Storage
    try:
        logger.info(f"Initializing HDF5 storage: {HDF5_FILE_NAME}...")
        _initialize_hdf5() # Ensure file and basic structure exists
        logger.info("HDF5 storage initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize HDF5 storage. Aborting.")
        # Depending on severity, might choose to exit or let MCP server handle it if it can run without storage.
        # For this service, storage is critical.
        return # Exit main if storage fails

    # 2. Configure Gemini AI Model
    try:
        logger.info("Configuring Gemini AI model...")
        ensure_gemini_configured()
        logger.info("Gemini AI model configured successfully.")
    except Exception as e:
        logger.exception("Failed to configure Gemini AI model. Queries requiring AI will fail.")
        # Decide if this is fatal. The server might still be able to serve stored data
        # that doesn't require AI. For now, log and continue.
        # If Gemini is absolutely essential for all operations, you might exit here.
        pass # Logged the error, continue to allow server to start for non-AI operations if any.

    # 3. Start the MCP Server
    logger.info("Starting MCP server...")
    try:
        # mcp_server.run() is blocking and starts the server.
        # It typically takes arguments for host, port, etc., but defaults are often fine for local dev.
        # Example: mcp_server.run(host="0.0.0.0", port=8000)
        # For now, using default run()
        mcp_server.run()
        logger.info("MCP server stopped.") # This line will be reached when server shuts down
    except Exception as e:
        logger.exception("MCP server failed to start or crashed.")

if __name__ == "__main__":
    main()
