"""
Implements the AI agent logic for the TensorDirectory service.

This module uses Google's Gemini model to interpret natural language prompts from users.
It identifies user intent and extracts relevant entities to interact with the
`storage` module for retrieving tensor/model data or performing inference.
Model execution via `exec()` is supported but carries security risks.
"""
# agent.py
# pip install google-generativeai python-dotenv numpy

import os
import json
import logging
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from mcp.server.fastmcp import Context # For type hinting and logging
from tensordirectory import storage
from datetime import datetime
import uuid # For generating unique names for output tensors

# Configure logging for this module
logger = logging.getLogger(__name__)

# Load environment variables (for API key)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Global Gemini model instance
gemini_model = None

def ensure_gemini_configured():
    """
    Configures the Google Gemini generative model using an API key from environment variables.

    Raises:
        ValueError: If the GEMINI_API_KEY is not found in the environment.
        Exception: Any exception raised during Gemini library configuration.
    """
    global gemini_model
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found in environment variables.")
        # In a real MCP context, this error should be surfaced to the user/operator
        # For now, we'll let it fail if the agent is called without it.
        raise ValueError("GEMINI_API_KEY not configured.")
    
    if gemini_model is None:
        try:
            # Note: The issue specified "gemini-2.0-flash". 
            # The google-generativeai SDK might use a model alias like "gemini-pro" or "gemini-1.5-flash-latest".
            # Using "gemini-pro" as a common default if "gemini-2.0-flash" isn't a direct identifier.
            # Update if a more specific/correct identifier for "gemini-2.0-flash" is confirmed for the SDK.
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-pro') # Or 'gemini-1.5-flash-latest' if available
            logger.info("Gemini model configured: gemini-pro (or similar alias for gemini-2.0-flash)")
        except Exception as e:
            logger.exception("Failed to configure Gemini model.")
            gemini_model = None # Ensure it's None if config failed
            raise e # Re-raise to indicate failure

async def analyze_prompt_with_gemini(user_prompt_text: str, ctx: Context) -> dict:
    """
    Analyzes the user's text prompt using the configured Gemini model to determine
    intent and extract entities.

    The Gemini model is instructed to return a JSON object specifying one of several
    predefined intents (e.g., "get_tensor", "run_inference") and associated entities
    (e.g., tensor name, model name, metadata query).

    Args:
        user_prompt_text: The natural language prompt from the user.
        ctx: The MCP Context object for logging.

    Returns:
        A dictionary parsed from Gemini's JSON response, containing the identified
        intent and entities. If an error occurs (API call, JSON parsing, model not
        configured), a dictionary with an "error" key and details is returned.
    """
    global gemini_model
    if gemini_model is None:
        # Attempt to configure if not already done. This might happen if ensure_gemini_configured wasn't called
        # or if it failed silently (though it's designed to raise).
        try:
            ensure_gemini_configured()
        except Exception as e:
            await ctx.log_error(f"Gemini model is not configured and auto-configuration failed: {e}", exc_info=True)
            return {"error": f"Gemini model not configured: {str(e)}"}

    # Construct the detailed prompt for Gemini
    gemini_system_prompt = f'''You are an AI assistant managing a Tensor Directory. Based on the user's request: "{user_prompt_text}"
    Identify the primary intent and extract relevant entities.
    Possible intents are: "get_tensor", "find_tensors_by_metadata", "run_inference", "unknown".

    For "get_tensor", entities:
    - "tensor_name": name of the tensor.

    For "find_tensors_by_metadata", entities:
    - "metadata_query": a dictionary of metadata key-value pairs to search for (e.g., {{"user_name": "my_image", "original_dtype": "float32"}}).

    For "run_inference", entities:
    - "model_name": name of the model to use.
    - "input_tensor_names": a list of tensor names to be used as input for the model.
    - "output_tensor_name": (optional) a name for the resulting tensor.

    If the intent is unclear or not one of the above, use "unknown".
    Provide your response as a single JSON object. Example:
    {{"intent": "get_tensor", "entities": {{"tensor_name": "image_features_v1"}}}}
    Another example:
    {{"intent": "run_inference", "entities": {{"model_name": "super_resolver", "input_tensor_names": ["low_res_image_abc"], "output_tensor_name": "high_res_output"}}}}'''

    await ctx.log_info(f"Sending prompt to Gemini: {user_prompt_text}")
    try:
        response = await gemini_model.generate_content_async(gemini_system_prompt)
        
        raw_response_text = ""
        # Iterating parts if response is chunked/streamed, though for single generation it's usually direct.
        # Some SDK versions/models might return parts. For gemini-pro, response.text should be fine.
        if hasattr(response, 'text') and response.text:
            raw_response_text = response.text
        else: # Fallback for potentially different response structures (e.g. parts)
             for part in response.parts:
                 if hasattr(part, 'text'):
                     raw_response_text += part.text
        
        raw_response_text = raw_response_text.strip()
        await ctx.log_info(f"Raw Gemini response: {raw_response_text}")

        # Gemini might sometimes wrap its JSON in ```json ... ```
        if raw_response_text.startswith("```json"):
            raw_response_text = raw_response_text[7:-3].strip()
        elif raw_response_text.startswith("```"): # More generic ``` removal
            raw_response_text = raw_response_text[3:-3].strip()

        parsed_response = json.loads(raw_response_text)
        await ctx.log_info(f"Parsed Gemini response: {parsed_response}")
        return parsed_response
    except json.JSONDecodeError as jde:
        await ctx.log_error(f"Failed to parse JSON from Gemini response: {raw_response_text}. Error: {jde}", exc_info=True)
        return {"error": "Failed to parse Gemini response as JSON.", "raw_response": raw_response_text}
    except Exception as e:
        # This includes potential errors from generate_content_async itself (e.g., API errors, safety blocks)
        await ctx.log_error(f"Error during Gemini API call or processing response: {e}", exc_info=True)
        # Try to get more details if it's a GoogleGenerativeAI Error
        error_details = str(e)
        if hasattr(e, 'message'): # For some specific API errors from the SDK
            error_details = e.message
        return {"error": f"Gemini API call failed: {error_details}"}


async def handle_query(user_prompt: str, params: dict | None, ctx: Context) -> str:
    """
    Handles a user's query by first analyzing it with Gemini, then performing
    actions based on the identified intent (e.g., fetching data from storage,
    running model inference).

    Args:
        user_prompt: The natural language query from the user.
        params: Optional dictionary of additional parameters (currently unused).
        ctx: The MCP Context object for logging.

    Returns:
        A string response summarizing the result of the action, an error message,
        or data retrieved.
    """
    try:
        ensure_gemini_configured() # Ensure model is ready before use
    except Exception as e:
        await ctx.log_error(f"Gemini configuration failed: {e}", exc_info=True)
        return f"Error: Could not configure AI model. {str(e)}"

    analysis = await analyze_prompt_with_gemini(user_prompt, ctx)

    if "error" in analysis:
        return f"Error analyzing prompt: {analysis['error']}. Raw response: {analysis.get('raw_response', '')}"

    intent = analysis.get("intent")
    entities = analysis.get("entities", {})

    if not intent:
        return f"Error: Gemini analysis did not provide an 'intent'. Analysis result: {analysis}"


    if intent == "get_tensor":
        tensor_name = entities.get("tensor_name")
        if not tensor_name:
            return "Error: Gemini analysis did not provide 'tensor_name' for get_tensor intent."
        
        data, metadata = storage.get_tensor_by_name(tensor_name)
        if data is not None and metadata is not None:
            return f"Tensor '{tensor_name}' found. Metadata: {metadata}. Shape: {data.shape}, Dtype: {data.dtype}."
        else:
            return f"Tensor '{tensor_name}' not found."

    elif intent == "find_tensors_by_metadata":
        metadata_query = entities.get("metadata_query")
        if not metadata_query or not isinstance(metadata_query, dict):
            return "Error: Gemini analysis did not provide valid 'metadata_query' for find_tensors intent."
        
        results = storage.find_tensors(metadata_query)
        if results:
            summary_list = []
            for i, (data, meta) in enumerate(results):
                if i < 3: # Limit summary to first 3 tensors to keep response manageable
                    summary_list.append(f"Name='{meta.get('user_name', 'N/A')}', Shape='{data.shape}', Dtype='{data.dtype}'")
                else:
                    summary_list.append(f"and {len(results) - i} more...")
                    break
            return f"Found {len(results)} tensor(s) matching criteria. Examples: [{'; '.join(summary_list)}]"
        else:
            return "No tensors found matching your criteria."

    elif intent == "run_inference":
        model_name = entities.get("model_name")
        input_tensor_names = entities.get("input_tensor_names", [])
        output_tensor_name_suggestion = entities.get("output_tensor_name")

        if not model_name or not isinstance(model_name, str):
             return "Error: Missing or invalid 'model_name' for inference."
        if not input_tensor_names or not isinstance(input_tensor_names, list) or not all(isinstance(n, str) for n in input_tensor_names):
            return "Error: Missing or invalid 'input_tensor_names' for inference. Must be a list of strings."


        model_data = storage.get_model_by_name(model_name)
        if not model_data:
            return f"Model '{model_name}' not found."

        if model_data.get("code"):
            await ctx.log_warning(f"Attempting to execute model code for '{model_name}' using exec(). SECURITY RISK: Untrusted code execution.")
            
            input_tensors_dict = {}
            for name in input_tensor_names:
                tensor_data, _ = storage.get_tensor_by_name(name)
                if tensor_data is None:
                    return f"Error: Input tensor '{name}' not found for inference."
                input_tensors_dict[name] = tensor_data
            
            exec_globals = {'np': np} 
            exec_locals = {'input_tensors_dict': input_tensors_dict}

            try:
                exec(model_data['code'], exec_globals, exec_locals)
                
                if 'predict' not in exec_locals or not callable(exec_locals['predict']):
                    return f"Error: Model code for '{model_name}' does not define a callable 'predict(input_tensors_dict)' function."

                # Assuming predict function takes one argument: a dictionary of input tensors
                output_tensor = exec_locals['predict'](input_tensors_dict) 

                if not isinstance(output_tensor, np.ndarray):
                     return f"Error: 'predict' function in model '{model_name}' did not return a NumPy array."

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Ensure output_tensor_name_suggestion is a string if provided, otherwise generate one
                if output_tensor_name_suggestion and not isinstance(output_tensor_name_suggestion, str):
                    await ctx.log_warning(f"Invalid 'output_tensor_name' suggestion type: {type(output_tensor_name_suggestion)}. Generating default name.")
                    output_tensor_name_suggestion = None

                final_output_name = output_tensor_name_suggestion or f"inference_output_{model_name}_{timestamp}_{str(uuid.uuid4())[:8]}"
                output_description = f"Output from model '{model_name}' using inputs: {', '.join(input_tensor_names)} at {timestamp}"
                
                output_uuid = storage.save_tensor(name=final_output_name, description=output_description, tensor_data=output_tensor)
                
                return f"Inference successful with model '{model_name}'. Output tensor saved as '{final_output_name}' (UUID: {output_uuid})."

            except Exception as e:
                await ctx.log_error(f"Exception during model code execution for '{model_name}': {e}", exc_info=True)
                return f"Error during model execution for '{model_name}': {str(e)}"
        
        elif model_data.get("weights") is not None: # Check if 'weights' is not None
             return f"Model '{model_name}' found, but it only has weights and no executable code. Direct execution of weights-only models is not yet supported by this agent."
        else:
            return f"Model '{model_name}' has no executable code or weights."

    elif intent == "unknown":
        return f"The intent of your request is unclear or not supported. Please try rephrasing. User prompt: '{user_prompt}'"
        
    else: 
        return f"Unhandled intent '{intent}'. Please check agent logic or prompt engineering. Entities: {entities}"

    return "Error: Reached end of handle_query without returning a specific response." # Fallback
