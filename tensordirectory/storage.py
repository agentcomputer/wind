"""
Manages the persistent storage of tensors and models using HDF5 files.

This module provides functions to:
- Initialize the HDF5 file structure.
- Save and retrieve tensors with their metadata.
- Save and retrieve models (Python code and/or NumPy weights) with metadata.
- Query tensors based on metadata.
- Maintain indices for mapping user-defined names to unique UUIDs for tensors and models.

Error handling for file operations and data retrieval is included.
Uses h5py for HDF5 interaction and numpy for tensor data.
"""
# storage.py

# Ensure these dependencies are available in your environment:
# pip install numpy h5py

import h5py
import numpy as np
import uuid
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any # Ensure Dict and Any are imported

HDF5_FILE_NAME = "tensor_directory.hdf5"

# Basic structure initialization (implement this first)
def _initialize_hdf5():
    """
    Ensures the HDF5 file and its basic group structure exist.
    Creates mapping datasets if they don't exist.
    """
    with h5py.File(HDF5_FILE_NAME, 'a') as hf:
        if '/tensors' not in hf:
            hf.create_group('/tensors')
        if '/models' not in hf:
            hf.create_group('/models')
        if '/indices' not in hf:
            hf.create_group('/indices')

        # Use variable-length string dtype for storing name-UUID mappings
        string_dt = h5py.string_dtype(encoding='utf-8')

        if '/indices/tensor_name_to_uuid' not in hf:
            # Create an empty, resizable dataset to store (name, uuid) pairs
            hf.create_dataset('/indices/tensor_name_to_uuid',
                              shape=(0, 2),
                              maxshape=(None, 2),
                              dtype=string_dt,
                              chunks=True)

        if '/indices/model_name_to_uuid' not in hf:
            # Create an empty, resizable dataset to store (name, uuid) pairs
            hf.create_dataset('/indices/model_name_to_uuid',
                              shape=(0, 2),
                              maxshape=(None, 2),
                              dtype=string_dt,
                              chunks=True)

def _get_uuid_from_name(hf: h5py.File, item_type: str, name: str) -> Optional[str]:
    """
    Looks up an item's UUID by its user-defined name from the HDF5 index.

    Args:
        hf: An open HDF5 file object.
        item_type: Either "tensor" or "model".
        name: The user-defined name to search for.

    Returns:
        The UUID string if found, otherwise None.
    """
    if item_type == "tensor":
        mapping_path = '/indices/tensor_name_to_uuid'
    elif item_type == "model":
        mapping_path = '/indices/model_name_to_uuid'
    else:
        # TODO: Replace print with proper logging (e.g., logger.warning or logger.error)
        print(f"Warning: Invalid item_type '{item_type}' in _get_uuid_from_name.")
        return None

    if mapping_path not in hf:
        return None

    mappings = hf[mapping_path][:]
    for i in range(mappings.shape[0]):
        stored_name_raw = mappings[i, 0]
        stored_uuid_raw = mappings[i, 1]

        current_stored_name = stored_name_raw.decode('utf-8') if isinstance(stored_name_raw, bytes) else str(stored_name_raw)

        if current_stored_name == name:
            return stored_uuid_raw.decode('utf-8') if isinstance(stored_uuid_raw, bytes) else str(stored_uuid_raw)

    return None

# Implement tensor functions here
def save_tensor(name: str, description: str, tensor_data: np.ndarray) -> str:
    """
    Saves a tensor and its metadata to the HDF5 file.
    Updates the name-to-UUID mapping. If the name already exists, its UUID is updated.

    Args:
        name: User-defined name for the tensor.
        description: Description of the tensor.
        tensor_data: NumPy array containing the tensor data.

    Returns:
        The generated UUID for the tensor.
    """
    _initialize_hdf5()
    tensor_uuid = str(uuid.uuid4())

    with h5py.File(HDF5_FILE_NAME, 'a') as hf:
        # Save tensor data
        dataset_path = f'/tensors/{tensor_uuid}'
        if dataset_path in hf:
            del hf[dataset_path] # Should not happen with UUIDs, but good practice
        dset = hf.create_dataset(dataset_path, data=tensor_data)

        # Store metadata as attributes
        dset.attrs['user_name'] = name
        dset.attrs['description'] = description
        dset.attrs['creation_date'] = datetime.now().isoformat()
        dset.attrs['original_dtype'] = str(tensor_data.dtype)
        dset.attrs['original_shape'] = str(tensor_data.shape)

        # Update tensor_name_to_uuid mapping
        mapping_path = '/indices/tensor_name_to_uuid'
        mappings = hf[mapping_path]

        # Convert to list for easier manipulation
        mappings_list = list(mappings[:]) # Contains (name_raw, uuid_raw) tuples

        name_exists = False
        for i in range(len(mappings_list)):
            existing_name_raw, old_uuid_raw = mappings_list[i] # Get raw values

            # Decode existing_name_raw if it's bytes
            current_existing_name = existing_name_raw.decode('utf-8') if isinstance(existing_name_raw, bytes) else str(existing_name_raw)

            if current_existing_name == name: # Compare decoded str with input str
                # Update with input 'name' (already str) and new 'tensor_uuid' (already str)
                mappings_list[i] = (name, tensor_uuid)
                name_exists = True
                break

        if not name_exists:
            mappings_list.append((name, tensor_uuid))

        # Resize and write back
        if len(mappings_list) > mappings.shape[0] :
             mappings.resize((len(mappings_list), 2))

        # Ensure data is in NumPy array format before writing
        # This step can be tricky with variable length strings if not handled correctly.
        # h5py expects a NumPy array (or similar) for dataset writes.
        if mappings_list: # only write if there's data
            # Convert list of tuples of Python strings to a NumPy array of bytes/objects
            # then let h5py handle the conversion to its string_dt
            # Using object dtype temporarily for the conversion, as h5py's string_dt handles UTF-8.
            # Or, ensure all strings are encoded to bytes first if not using h5py.string_dtype directly for assignment
            string_dt = h5py.string_dtype(encoding='utf-8')
            # Create a new numpy array with the correct dtype for assignment
            # This conversion is important.
            new_mappings_arr = np.array(mappings_list, dtype=string_dt)
            hf[mapping_path][:] = new_mappings_arr
        elif mappings.shape[0] > 0 : # if list is empty but dataset wasn't
             hf[mapping_path].resize((0,2))


    return tensor_uuid

def get_tensor_by_uuid(uuid_str: str) -> tuple[np.ndarray | None, dict | None]:
    """
    Retrieves a tensor and its metadata by its UUID.

    Args:
        uuid_str: The UUID of the tensor to retrieve.

    Returns:
        A tuple (tensor_data, metadata_dict) or (None, None) if not found.
    """
    _initialize_hdf5() # Ensures file structure is known, though reading might not strictly need it
    try:
        with h5py.File(HDF5_FILE_NAME, 'r') as hf:
            dataset_path = f'/tensors/{uuid_str}'
            if dataset_path in hf:
                dset = hf[dataset_path]
                tensor_data = dset[:]
                metadata = dict(dset.attrs)
                # Add the UUID to the metadata dict for convenience for the caller
                metadata['uuid'] = uuid_str
                return tensor_data, metadata
            else:
                return None, None
    except FileNotFoundError:
        # This can happen if HDF5_FILE_NAME does not exist at all.
        # _initialize_hdf5 creates it, but if called on a system where it's then deleted
        # before this read, this is a possible path.
        return None, None
    except Exception as e:
        # TODO: Consider replacing print with logging or raising custom exceptions
        print(f"Error retrieving tensor by UUID {uuid_str}: {e}")
        return None, None

def get_tensor_by_name(name: str) -> tuple[np.ndarray | None, dict | None]:
    """
    Retrieves a tensor and its metadata by its user-defined name.

    Args:
        name: The user-defined name of the tensor.

    Returns:
        A tuple (tensor_data, metadata_dict) or (None, None) if not found.
    """
    _initialize_hdf5()
    found_uuid = None  # Initialize here

    try:
        with h5py.File(HDF5_FILE_NAME, 'r') as hf:
            mapping_path = '/indices/tensor_name_to_uuid'
            if mapping_path not in hf:
                # Index itself doesn't exist, found_uuid remains None
                pass
            else:
                mappings = hf[mapping_path][:]
                # Iterate by index to handle potential byte arrays from h5py
                for i in range(mappings.shape[0]):
                    stored_name_raw = mappings[i, 0]
                    stored_uuid_raw = mappings[i, 1]

                    # Decode if bytes, otherwise assume str
                    current_stored_name = stored_name_raw.decode('utf-8') if isinstance(stored_name_raw, bytes) else str(stored_name_raw)

                    if current_stored_name == name: # name is input Python str
                        found_uuid = stored_uuid_raw.decode('utf-8') if isinstance(stored_uuid_raw, bytes) else str(stored_uuid_raw)
                        break

        # ---- End of 'with' block for reading the index ----
        # Now hf (for index read) is closed. Proceed based on found_uuid.

        if found_uuid:
            # This call will open the file again, but the previous handle is closed.
            return get_tensor_by_uuid(found_uuid)
        else:
            return None, None

    except FileNotFoundError:
        return None, None
    except Exception as e:
        # TODO: Consider replacing print with logging or raising custom exceptions
        print(f"Error retrieving tensor by name {name}: {e}")
        return None, None

def find_tensors(metadata_query: dict) -> list[tuple[np.ndarray, dict]]:
    """
    Finds tensors based on matching metadata criteria.

    Args:
        metadata_query: A dictionary where keys are metadata attribute names
                        and values are the desired values.

    Returns:
        A list of (tensor_data, metadata_dict) tuples for matching tensors.
    """
    _initialize_hdf5()
    matches = []
    try:
        with h5py.File(HDF5_FILE_NAME, 'r') as hf:
            if '/tensors' not in hf:
                return [] # No tensors group

            tensors_group = hf['/tensors']
            for uuid_str in tensors_group:
                dset = tensors_group[uuid_str]
                metadata = dict(dset.attrs)

                match = True
                for query_key, query_value in metadata_query.items():
                    if metadata.get(query_key) != query_value:
                        match = False
                        break

                if match:
                    tensor_data = dset[:]
                    matches.append((tensor_data, metadata))
        return matches
    except FileNotFoundError:
        return [] # File doesn't exist
    except Exception as e:
        # TODO: Consider replacing print with logging or raising custom exceptions
        print(f"Error finding tensors: {e}")
        return []


# Implement model functions here
def save_model(name: str, description: str, model_weights: np.ndarray | None = None, model_code: str | None = None) -> str:
    """
    Saves a model (weights and/or code) and its metadata to the HDF5 file.
    Updates the model name-to-UUID mapping.

    Args:
        name: User-defined name for the model.
        description: Description of the model.
        model_weights: Optional NumPy array containing model weights.
        model_code: Optional string containing model code.

    Returns:
        The generated UUID for the model.
    """
    _initialize_hdf5()
    model_uuid = str(uuid.uuid4())

    with h5py.File(HDF5_FILE_NAME, 'a') as hf:
        model_group_path = f'/models/{model_uuid}'
        if model_group_path in hf:
            del hf[model_group_path] # Should not happen, but good practice
        model_group = hf.create_group(model_group_path)

        # Store metadata as attributes of the group
        model_group.attrs['user_name'] = name
        model_group.attrs['description'] = description
        model_group.attrs['upload_date'] = datetime.now().isoformat()

        # Save model weights if provided
        if model_weights is not None:
            model_group.create_dataset('weights', data=model_weights)

        # Save model code if provided
        if model_code is not None:
            # Use variable-length string type for code
            string_dt = h5py.string_dtype(encoding='utf-8')
            model_group.create_dataset('code', data=model_code, dtype=string_dt)

        # Update model_name_to_uuid mapping
        mapping_path = '/indices/model_name_to_uuid'
        mappings = hf[mapping_path]
        mappings_list = list(mappings[:]) # Contains (name_raw, uuid_raw) tuples

        name_exists = False
        for i in range(len(mappings_list)):
            existing_name_raw, old_uuid_raw = mappings_list[i] # Get raw values

            # Decode existing_name_raw if it's bytes
            current_existing_name = existing_name_raw.decode('utf-8') if isinstance(existing_name_raw, bytes) else str(existing_name_raw)

            if current_existing_name == name: # Compare decoded str with input str
                # Update with input 'name' (already str) and new 'model_uuid' (already str)
                mappings_list[i] = (name, model_uuid)
                name_exists = True
                break

        if not name_exists:
            mappings_list.append((name, model_uuid))

        if len(mappings_list) > mappings.shape[0]:
            mappings.resize((len(mappings_list), 2))

        if mappings_list:
            string_dt = h5py.string_dtype(encoding='utf-8')
            new_mappings_arr = np.array(mappings_list, dtype=string_dt)
            hf[mapping_path][:] = new_mappings_arr
        elif mappings.shape[0] > 0:
            hf[mapping_path].resize((0,2))

    return model_uuid

def get_model_by_uuid(uuid_str: str) -> dict | None:
    """
    Retrieves model data (weights, code) and metadata by its UUID.

    Args:
        uuid_str: The UUID of the model to retrieve.

    Returns:
        A dictionary {'metadata': {...}, 'weights': np.ndarray | None, 'code': str | None}
        or None if not found.
    """
    _initialize_hdf5()
    try:
        with h5py.File(HDF5_FILE_NAME, 'r') as hf:
            model_group_path = f'/models/{uuid_str}'
            if model_group_path not in hf:
                return None

            model_group = hf[model_group_path]
            metadata = dict(model_group.attrs)

            model_weights = None
            if 'weights' in model_group:
                model_weights = model_group['weights'][:]

            model_code = None
            if 'code' in model_group:
                # Data is stored as bytes, needs decoding if it's not auto-handled by string_dt read
                code_data = model_group['code'][()] # Read scalar/string data
                if isinstance(code_data, bytes):
                     model_code = code_data.decode('utf-8')
                else: # Should be a string if stored with h5py.string_dtype
                    model_code = code_data

            # Add the UUID to the metadata dict for convenience
            metadata['uuid'] = uuid_str

            return {
                'metadata': metadata,
                'weights': model_weights,
                'code': model_code
            }
    except FileNotFoundError:
        return None
    except Exception as e:
        # TODO: Consider replacing print with logging or raising custom exceptions
        print(f"Error retrieving model by UUID {uuid_str}: {e}")
        return None

def get_model_by_name(name: str) -> dict | None:
    """
    Retrieves model data and metadata by its user-defined name.

    Args:
        name: The user-defined name of the model.

    Returns:
        A dictionary {'metadata': {...}, 'weights': np.ndarray | None, 'code': str | None}
        or None if not found.
    """
    _initialize_hdf5()
    found_uuid = None  # Initialize here

    try:
        with h5py.File(HDF5_FILE_NAME, 'r') as hf:
            mapping_path = '/indices/model_name_to_uuid'
            if mapping_path not in hf:
                # Index itself doesn't exist, found_uuid remains None
                pass
            else:
                mappings = hf[mapping_path][:]
                # Iterate by index to handle potential byte arrays from h5py
                for i in range(mappings.shape[0]):
                    stored_name_raw = mappings[i, 0]
                    stored_uuid_raw = mappings[i, 1]

                    # Decode if bytes, otherwise assume str
                    current_stored_name = stored_name_raw.decode('utf-8') if isinstance(stored_name_raw, bytes) else str(stored_name_raw)

                    if current_stored_name == name: # name is input Python str
                        found_uuid = stored_uuid_raw.decode('utf-8') if isinstance(stored_uuid_raw, bytes) else str(stored_uuid_raw)
                        break

        # ---- End of 'with' block for reading the index ----
        # Now hf (for index read) is closed. Proceed based on found_uuid.

        if found_uuid:
            # This call will open the file again, but the previous handle is closed.
            return get_model_by_uuid(found_uuid)
        else:
            return None

    except FileNotFoundError:
        return None
    except Exception as e:
        # TODO: Consider replacing print with logging or raising custom exceptions
        print(f"Error retrieving model by name {name}: {e}")
        return None

# --- Helper for index removal ---
def _remove_uuid_from_index(hf: h5py.File, index_path: str, uuid_to_remove: str) -> bool:
    """
    Removes all entries associated with a specific UUID from a given index dataset.

    Args:
        hf: An open HDF5 file object (opened in 'a' mode).
        index_path: The HDF5 path to the name-to-UUID mapping dataset.
        uuid_to_remove: The UUID string whose entries should be removed.

    Returns:
        True if the operation was successful (even if UUID was not found), False on error.
    """
    if index_path not in hf:
        # TODO: Replace print with proper logging
        print(f"Warning: Index path '{index_path}' not found during UUID removal.")
        return False # Or True, as there's nothing to remove. Let's say False for path not found.

    try:
        mappings = hf[index_path][:]
        # Filter out entries where the UUID matches uuid_to_remove
        # Need to decode stored UUID if it's bytes, for comparison with uuid_to_remove (str)
        updated_mappings_list = [
            (name_raw, uuid_raw) for name_raw, uuid_raw in mappings
            if (uuid_raw.decode('utf-8') if isinstance(uuid_raw, bytes) else str(uuid_raw)) != uuid_to_remove
        ]

        current_shape_val = mappings.shape[0]
        new_shape_val = len(updated_mappings_list)

        if new_shape_val < current_shape_val or (current_shape_val > 0 and new_shape_val == 0) : # Check if any changes were made or if list became empty
            if updated_mappings_list:
                new_arr = np.array(updated_mappings_list, dtype=h5py.string_dtype(encoding='utf-8'))
                hf[index_path].resize((new_shape_val, 2))
                hf[index_path][:] = new_arr
            else: # List is now empty
                hf[index_path].resize((0, 2))
        return True
    except Exception as e:
        # TODO: Replace print with proper logging
        print(f"Error updating index path '{index_path}' for UUID '{uuid_to_remove}': {e}")
        return False

# --- Delete functions ---
def delete_tensor_by_uuid(uuid_str: str) -> bool:
    """
    Deletes a tensor dataset and its corresponding entries from the name-to-UUID index.

    Args:
        uuid_str: The UUID of the tensor to delete.

    Returns:
        True if the tensor was successfully deleted, False otherwise.
    """
    _initialize_hdf5() # Ensure basic paths exist, though not strictly needed if file must exist
    dataset_path = f'/tensors/{uuid_str}'
    deleted_dataset = False

    try:
        with h5py.File(HDF5_FILE_NAME, 'a') as hf:
            if dataset_path in hf:
                del hf[dataset_path]
                deleted_dataset = True
            else:
                # Dataset not found, but we might still want to clean the index if it exists there.
                # Consider if this is an error or just "not found, nothing to delete".
                # For now, if dataset doesn't exist, assume it's not an error, just nothing to delete from /tensors.
                pass # deleted_dataset remains False if we want to signal it wasn't found here.

            # Always attempt to clean the index for this UUID, even if dataset was already gone.
            # This handles cases where index might have stale entries.
            index_cleaned_successfully = _remove_uuid_from_index(hf, '/indices/tensor_name_to_uuid', uuid_str)

            # Return True if dataset was found and deleted.
            # If dataset wasn't found, it implies it was already deleted or never existed.
            # The function should signal if the primary deletion occurred.
            # If index cleaning fails, that's an internal issue but the primary data might be gone.
            # Let's return True if the dataset was confirmed deleted.
            return deleted_dataset

    except Exception as e:
        # TODO: Replace print with proper logging
        print(f"Error deleting tensor UUID '{uuid_str}': {e}")
        return False

def delete_model_by_uuid(uuid_str: str) -> bool:
    """
    Deletes a model group and its corresponding entries from the name-to-UUID index.

    Args:
        uuid_str: The UUID of the model to delete.

    Returns:
        True if the model was successfully deleted, False otherwise.
    """
    _initialize_hdf5()
    model_group_path = f'/models/{uuid_str}'
    deleted_group = False

    try:
        with h5py.File(HDF5_FILE_NAME, 'a') as hf:
            if model_group_path in hf:
                del hf[model_group_path]
                deleted_group = True
            else:
                pass # deleted_group remains False

            index_cleaned_successfully = _remove_uuid_from_index(hf, '/indices/model_name_to_uuid', uuid_str)

            return deleted_group

    except Exception as e:
        # TODO: Replace print with proper logging
        print(f"Error deleting model UUID '{uuid_str}': {e}")
        return False

# --- List functions ---
def list_tensors(filter_by_name_contains: Optional[str] = None, limit: int = 100, offset: int = 0) -> Tuple[List[dict], int]:
    """
    Lists tensors with optional filtering by name and pagination.

    Args:
        filter_by_name_contains: Substring to filter tensor names by (case-insensitive).
        limit: Maximum number of items to return.
        offset: Offset for pagination.

    Returns:
        A tuple containing:
            - A list of metadata dictionaries for the paginated tensors.
            - The total count of tensors matching the filter (before pagination).
    """
    _initialize_hdf5()
    all_matching_metadata = []

    try:
        with h5py.File(HDF5_FILE_NAME, 'r') as hf:
            if '/tensors' not in hf:
                return [], 0

            tensors_group = hf['/tensors']
            # Collect all names/UUIDs first to apply filter before loading all attrs,
            # though for attributes it's usually not a huge overhead.
            # If filtering on data itself, more care would be needed.

            candidate_uuids = list(tensors_group.keys()) # Get all UUIDs

            for uuid_str in candidate_uuids:
                dset = tensors_group[uuid_str]
                metadata = dict(dset.attrs)
                metadata['uuid'] = uuid_str # Ensure UUID is part of the returned metadata

                # Ensure 'user_name' exists, providing a default if somehow missing (should not happen)
                user_name = metadata.get('user_name', '')

                if filter_by_name_contains:
                    if filter_by_name_contains.lower() not in user_name.lower():
                        continue # Skip if name doesn't match filter

                all_matching_metadata.append(metadata)

        total_count = len(all_matching_metadata)
        # Sort by user_name for consistent pagination, can be made configurable
        all_matching_metadata.sort(key=lambda x: x.get('user_name', '').lower())
        paginated_results = all_matching_metadata[offset : offset + limit]

        return paginated_results, total_count

    except FileNotFoundError:
        return [], 0
    except Exception as e:
        # TODO: Replace print with proper logging
        print(f"Error listing tensors: {e}")
        return [], 0

def list_models(filter_by_name_contains: Optional[str] = None, limit: int = 100, offset: int = 0) -> Tuple[List[dict], int]:
    """
    Lists models with optional filtering by name and pagination.

    Args:
        filter_by_name_contains: Substring to filter model names by (case-insensitive).
        limit: Maximum number of items to return.
        offset: Offset for pagination.

    Returns:
        A tuple containing:
            - A list of metadata dictionaries for the paginated models.
            - The total count of models matching the filter (before pagination).
    """
    _initialize_hdf5()
    all_matching_metadata = []

    try:
        with h5py.File(HDF5_FILE_NAME, 'r') as hf:
            if '/models' not in hf:
                return [], 0

            models_group = hf['/models']
            candidate_uuids = list(models_group.keys())

            for uuid_str in candidate_uuids:
                model_item_group = models_group[uuid_str] # Corrected variable name
                metadata = dict(model_item_group.attrs)
                metadata['uuid'] = uuid_str

                user_name = metadata.get('user_name', '')

                if filter_by_name_contains:
                    if filter_by_name_contains.lower() not in user_name.lower():
                        continue

                metadata['has_code'] = 'code' in model_item_group
                metadata['has_weights'] = 'weights' in model_item_group
                all_matching_metadata.append(metadata)

        total_count = len(all_matching_metadata)
        all_matching_metadata.sort(key=lambda x: x.get('user_name', '').lower())
        paginated_results = all_matching_metadata[offset : offset + limit]

        return paginated_results, total_count

    except FileNotFoundError:
        return [], 0
    except Exception as e:
        # TODO: Replace print with proper logging
        print(f"Error listing models: {e}")
        return [], 0

# --- Helper for updating name in index ---
def _update_name_in_index(hf: h5py.File, index_path: str, old_user_name: Optional[str], new_user_name: str, item_uuid: str) -> bool:
    """
    Updates or sets an item's name in the specified index.
    It ensures that the new_user_name correctly points to item_uuid and that
    no other items point to new_user_name. It also removes the old_user_name mapping
    for this specific item_uuid if old_user_name is provided.

    Args:
        hf: An open HDF5 file object (opened in 'a' mode).
        index_path: The HDF5 path to the name-to-UUID mapping dataset.
        old_user_name: The previous user_name of the item. Can be None.
        new_user_name: The new user_name for the item. Must be a non-empty string.
        item_uuid: The UUID of the item whose name is being updated.

    Returns:
        True if the index was updated successfully, False otherwise.
    """
    if not new_user_name: # new_user_name must not be empty for an update that sets a name
        # TODO: Proper logging
        print(f"Error: New user_name cannot be empty for UUID {item_uuid} in index {index_path}.")
        return False # Or raise ValueError

    if index_path not in hf:
        # This case should ideally be handled by _initialize_hdf5, but defensive check here.
        # TODO: Proper logging
        print(f"Error: Index path '{index_path}' not found during name update.")
        return False

    try:
        mappings_list = list(hf[index_path][:])
        final_mappings = []
        processed_new_name_for_item = False

        for name_raw, uuid_raw in mappings_list:
            current_name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else str(name_raw)
            current_uuid = uuid_raw.decode('utf-8') if isinstance(uuid_raw, bytes) else str(uuid_raw)

            # Case 1: Entry for the same item_uuid but with the old_name. Remove it.
            if old_user_name and current_name == old_user_name and current_uuid == item_uuid:
                continue

            # Case 2: Any existing entry with the new_user_name. Remove it, as new_user_name will now point to item_uuid.
            if current_name == new_user_name:
                continue

            # Case 3: If this entry is for the item_uuid but not the old_name (e.g. another alias, though not supported yet)
            # or if it's for a completely different item, keep it.
            final_mappings.append((name_raw, uuid_raw))

        # Add the new mapping for the item.
        final_mappings.append((new_user_name.encode('utf-8'), item_uuid.encode('utf-8')))

        # Write back the updated mappings
        if final_mappings:
            new_arr = np.array(final_mappings, dtype=h5py.string_dtype(encoding='utf-8'))
            hf[index_path].resize((len(final_mappings), 2))
            hf[index_path][:] = new_arr
        else:
            hf[index_path].resize((0, 2))
        return True

    except Exception as e:
        # TODO: Replace print with proper logging
        print(f"Error updating name in index '{index_path}' for item '{item_uuid}': {e}")
        return False


# --- Metadata Update functions ---
def update_tensor_metadata(uuid_str: str, metadata_updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Updates the metadata for a given tensor UUID.
    Currently supports updating 'user_name' and 'description'.
    If 'user_name' is updated, the name-to-UUID index is also updated.

    Args:
        uuid_str: The UUID of the tensor to update.
        metadata_updates: A dictionary containing the metadata fields to update
                          and their new values. e.g., {"user_name": "new_name", "description": "new_desc"}

    Returns:
        The full updated metadata dictionary if successful, None otherwise.
    """
    _initialize_hdf5()
    dataset_path = f'/tensors/{uuid_str}'

    try:
        with h5py.File(HDF5_FILE_NAME, 'a') as hf:
            if dataset_path not in hf:
                return None # Tensor not found

            dset = hf[dataset_path]
            old_user_name = dset.attrs.get('user_name')
            name_changed = False
            new_user_name = old_user_name

            for key, value in metadata_updates.items():
                if key == 'user_name':
                    if isinstance(value, str) and value.strip(): # Ensure new name is a non-empty string
                        potential_new_name = value.strip()
                        if potential_new_name != old_user_name:
                            new_user_name = potential_new_name
                            dset.attrs['user_name'] = new_user_name
                            name_changed = True
                    else:
                        # TODO: Log warning about invalid user_name update attempt
                        print(f"Warning: Invalid 'user_name' update value for tensor {uuid_str}: {value}")
                elif key == 'description':
                    if isinstance(value, str): # Allow empty string for description
                        dset.attrs['description'] = value
                    else:
                        # TODO: Log warning
                        print(f"Warning: Invalid 'description' update value for tensor {uuid_str}: {value}")
                # Add other updatable attributes here if needed
                # else:
                #     logger.warning(f"Attempt to update unsupport attribute '{key}' for tensor {uuid_str}")


            if name_changed:
                if not _update_name_in_index(hf, '/indices/tensor_name_to_uuid', old_user_name, new_user_name, uuid_str):
                    # TODO: Handle index update failure - potentially revert attribute change or log critical error
                    print(f"Critical: Failed to update index for tensor {uuid_str} after name change from '{old_user_name}' to '{new_user_name}'.")
                    # For now, we proceed, but this indicates a potential inconsistency.

            updated_metadata = dict(dset.attrs)
            updated_metadata['uuid'] = uuid_str # Ensure UUID is in the returned metadata
            return updated_metadata

    except Exception as e:
        # TODO: Replace print with proper logging
        print(f"Error updating tensor metadata for UUID '{uuid_str}': {e}")
        return None

def update_model_metadata(uuid_str: str, metadata_updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Updates the metadata for a given model UUID.
    Currently supports updating 'user_name' and 'description'.
    If 'user_name' is updated, the name-to-UUID index is also updated.

    Args:
        uuid_str: The UUID of the model to update.
        metadata_updates: A dictionary containing the metadata fields to update
                          and their new values. e.g., {"user_name": "new_name", "description": "new_desc"}

    Returns:
        The full updated metadata dictionary if successful, None otherwise.
    """
    _initialize_hdf5()
    model_group_path = f'/models/{uuid_str}'

    try:
        with h5py.File(HDF5_FILE_NAME, 'a') as hf:
            if model_group_path not in hf:
                return None # Model not found

            model_group = hf[model_group_path]
            old_user_name = model_group.attrs.get('user_name')
            name_changed = False
            new_user_name = old_user_name

            for key, value in metadata_updates.items():
                if key == 'user_name':
                    if isinstance(value, str) and value.strip():
                        potential_new_name = value.strip()
                        if potential_new_name != old_user_name:
                            new_user_name = potential_new_name
                            model_group.attrs['user_name'] = new_user_name
                            name_changed = True
                    else:
                        print(f"Warning: Invalid 'user_name' update value for model {uuid_str}: {value}")
                elif key == 'description':
                    if isinstance(value, str):
                        model_group.attrs['description'] = value
                    else:
                        print(f"Warning: Invalid 'description' update value for model {uuid_str}: {value}")

            if name_changed:
                if not _update_name_in_index(hf, '/indices/model_name_to_uuid', old_user_name, new_user_name, uuid_str):
                    print(f"Critical: Failed to update index for model {uuid_str} after name change from '{old_user_name}' to '{new_user_name}'.")

            updated_metadata = dict(model_group.attrs)
            updated_metadata['uuid'] = uuid_str
            updated_metadata['has_code'] = 'code' in model_group
            updated_metadata['has_weights'] = 'weights' in model_group
            return updated_metadata

    except Exception as e:
        # TODO: Replace print with proper logging
        print(f"Error updating model metadata for UUID '{uuid_str}': {e}")
        return None
