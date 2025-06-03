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
