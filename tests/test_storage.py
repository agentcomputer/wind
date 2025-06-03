# tests/test_storage.py
import unittest
import os
import numpy as np
import h5py
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional # Ensure all needed types are imported

# Adjust import path if necessary, assuming 'tensordirectory' is a package
# and tests are run from the project root or with PYTHONPATH set.
from tensordirectory import storage

TEST_HDF5_FILE = "test_tensor_directory.hdf5"

class TestStorage(unittest.TestCase):

    def setUp(self):
        # Ensure a clean slate for each test
        if os.path.exists(TEST_HDF5_FILE):
            os.remove(TEST_HDF5_FILE)
        storage.HDF5_FILE_NAME = TEST_HDF5_FILE
        # We call _initialize_hdf5() in each test that needs a fresh file,
        # or here if all tests expect it to be pre-initialized.
        # For clarity, often it's better to let each test manage its specific setup if needed,
        # but for this structure, initializing here is fine as most tests will need it.
        storage._initialize_hdf5()

    def tearDown(self):
        # Clean up the test file after each test
        if os.path.exists(TEST_HDF5_FILE):
            os.remove(TEST_HDF5_FILE)
        # Reset HDF5_FILE_NAME if it's a global in storage that might affect other tests/modules
        # This simple assignment works because Python modules are singletons.
        # If storage.py was more complex, might need a more robust way to reset.
        # For now, we assume storage.py will re-evaluate its HDF5_FILE_NAME on next import or use,
        # or that this test suite is the only user during its run.
        # A safer way would be to store the original name and restore it.
        # However, the prompt uses `storage.HDF5_FILE_NAME = "tensor_directory.hdf5"` in the module,
        # so we might need to reset it to that. For this test suite, it's okay.
        # Let's assume the original name is "tensor_directory.hdf5" as per storage.py
        storage.HDF5_FILE_NAME = "tensor_directory.hdf5"


    def test_01_initialize_hdf5(self):
        # setUp already calls _initialize_hdf5()
        self.assertTrue(os.path.exists(TEST_HDF5_FILE))
        with h5py.File(TEST_HDF5_FILE, 'r') as f:
            self.assertIn('tensors', f)
            self.assertIn('models', f)
            self.assertIn('indices', f)
            self.assertIn('indices/tensor_name_to_uuid', f)
            self.assertIsInstance(f['indices/tensor_name_to_uuid'], h5py.Dataset)
            self.assertEqual(f['indices/tensor_name_to_uuid'].shape, (0,2))
            self.assertIn('indices/model_name_to_uuid', f)
            self.assertIsInstance(f['indices/model_name_to_uuid'], h5py.Dataset)
            self.assertEqual(f['indices/model_name_to_uuid'].shape, (0,2))


    def test_02_save_and_get_tensor(self):
        tensor_name = "test_tensor_1"
        description = "A test tensor"
        data = np.array([[1, 2], [3, 4]], dtype=np.int32)

        uuid_val = storage.save_tensor(name=tensor_name, description=description, tensor_data=data)
        self.assertIsInstance(uuid_val, str)

        # Get by UUID
        ret_data_uuid, ret_meta_uuid = storage.get_tensor_by_uuid(uuid_val)
        self.assertIsNotNone(ret_data_uuid)
        self.assertIsNotNone(ret_meta_uuid)
        np.testing.assert_array_equal(ret_data_uuid, data)
        self.assertEqual(ret_meta_uuid['user_name'], tensor_name)
        self.assertEqual(ret_meta_uuid['description'], description)
        self.assertEqual(ret_meta_uuid['original_dtype'], str(data.dtype))
        self.assertEqual(ret_meta_uuid['original_shape'], str(data.shape))
        self.assertTrue('creation_date' in ret_meta_uuid)

        # Get by Name
        ret_data_name, ret_meta_name = storage.get_tensor_by_name(tensor_name)
        self.assertIsNotNone(ret_data_name)
        self.assertIsNotNone(ret_meta_name)
        np.testing.assert_array_equal(ret_data_name, data)
        self.assertEqual(ret_meta_name['user_name'], tensor_name)
        self.assertEqual(ret_meta_name['uuid'], uuid_val) # Check if metadata from get_by_name also includes uuid

    def test_03_get_tensor_not_found(self):
        ret_data, ret_meta = storage.get_tensor_by_uuid("non_existent_uuid")
        self.assertIsNone(ret_data)
        self.assertIsNone(ret_meta)

        ret_data_name, ret_meta_name = storage.get_tensor_by_name("non_existent_name")
        self.assertIsNone(ret_data_name)
        self.assertIsNone(ret_meta_name)

    def test_04_save_tensor_name_overwrite(self):
        name = "overwrite_tensor"
        data1 = np.array([1, 2, 3], dtype=np.float64)
        data2 = np.array([4, 5, 6], dtype=np.int16)

        uuid1 = storage.save_tensor(name=name, description="first_version", tensor_data=data1)
        ret_data1_name, meta1_name = storage.get_tensor_by_name(name)
        np.testing.assert_array_equal(ret_data1_name, data1)
        self.assertEqual(meta1_name['description'], "first_version")

        uuid2 = storage.save_tensor(name=name, description="second_version", tensor_data=data2)
        self.assertNotEqual(uuid1, uuid2)

        ret_data2_name, meta2_name = storage.get_tensor_by_name(name)
        np.testing.assert_array_equal(ret_data2_name, data2)
        self.assertEqual(meta2_name['description'], "second_version")
        self.assertEqual(meta2_name['original_dtype'], str(data2.dtype))

        # Old UUID should still fetch the old data
        ret_data_uuid1, meta_uuid1 = storage.get_tensor_by_uuid(uuid1)
        np.testing.assert_array_equal(ret_data_uuid1, data1)
        self.assertEqual(meta_uuid1['description'], "first_version")

        # New UUID from name lookup should match uuid2
        self.assertEqual(meta2_name['uuid'], uuid2)


    def test_05_find_tensors(self):
        storage.save_tensor("t1", "desc1_float_userA", np.array([1.0, 2.0], dtype=np.float32))
        storage.save_tensor("t2_userB", "desc2_int_userB", np.array([1, 2], dtype=np.int32))
        storage.save_tensor("t3_userA", "desc3_float_userA", np.array([3.0, 4.0], dtype=np.float32))

        results_float = storage.find_tensors({"original_dtype": "float32"})
        self.assertEqual(len(results_float), 2)
        found_names_float = sorted([meta['user_name'] for _, meta in results_float])
        self.assertListEqual(found_names_float, sorted(["t1", "t3_userA"]))

        results_int = storage.find_tensors({"original_dtype": "int32"})
        self.assertEqual(len(results_int), 1)
        self.assertEqual(results_int[0][1]['user_name'], "t2_userB")

        results_userA_float = storage.find_tensors({"original_dtype": "float32", "user_name": "t3_userA"})
        self.assertEqual(len(results_userA_float), 1)
        self.assertEqual(results_userA_float[0][1]['user_name'], "t3_userA")

        results_no_match = storage.find_tensors({"user_name": "non_existent_tensor"})
        self.assertEqual(len(results_no_match), 0)

        results_all = storage.find_tensors({}) # Empty query should return all
        self.assertEqual(len(results_all), 3)


    def test_06_tensor_data_types_and_shapes(self):
        test_cases = [
            ("int_1d", np.array([1, 2, 3], dtype=np.int32)),
            ("float_2d", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
            ("bool_3d", np.array([[[True, False], [False, True]]], dtype=np.bool_)),
            ("empty_array", np.array([], dtype=np.float64))
        ]
        for name, data in test_cases:
            with self.subTest(name=name):
                uuid_val = storage.save_tensor(name, f"Test for {name}", data)
                ret_data, ret_meta = storage.get_tensor_by_uuid(uuid_val)
                np.testing.assert_array_equal(ret_data, data)
                self.assertEqual(ret_meta['original_dtype'], str(data.dtype))
                self.assertEqual(ret_meta['original_shape'], str(data.shape))

    def test_07_save_and_get_model_variants(self):
        model_name_full = "test_model_full"
        desc_full = "Full model with code and weights"
        code_full = "def predict(data): return data * 2"
        weights_full = np.array([1.0, 2.0, 3.0])
        uuid_full = storage.save_model(model_name_full, desc_full, model_weights=weights_full, model_code=code_full)

        ret_model_full_uuid = storage.get_model_by_uuid(uuid_full)
        self.assertIsNotNone(ret_model_full_uuid)
        self.assertEqual(ret_model_full_uuid['metadata']['user_name'], model_name_full)
        self.assertEqual(ret_model_full_uuid['metadata']['description'], desc_full)
        np.testing.assert_array_equal(ret_model_full_uuid['weights'], weights_full)
        self.assertEqual(ret_model_full_uuid['code'], code_full)
        self.assertTrue('upload_date' in ret_model_full_uuid['metadata'])

        ret_model_full_name = storage.get_model_by_name(model_name_full)
        self.assertIsNotNone(ret_model_full_name)
        np.testing.assert_array_equal(ret_model_full_name['weights'], weights_full)
        self.assertEqual(ret_model_full_name['code'], code_full)
        self.assertEqual(ret_model_full_name['metadata']['uuid'], uuid_full)


        model_name_code_only = "test_model_code_only"
        desc_code_only = "Model with only code"
        code_code_only = "def transform(x): return x + 1"
        uuid_code_only = storage.save_model(model_name_code_only, desc_code_only, model_code=code_code_only)

        ret_model_code_only = storage.get_model_by_uuid(uuid_code_only)
        self.assertIsNotNone(ret_model_code_only)
        self.assertEqual(ret_model_code_only['metadata']['user_name'], model_name_code_only)
        self.assertEqual(ret_model_code_only['code'], code_code_only)
        self.assertIsNone(ret_model_code_only['weights'])

        model_name_weights_only = "test_model_weights_only"
        desc_weights_only = "Model with only weights"
        weights_only_data = np.array([10, 20])
        uuid_weights_only = storage.save_model(model_name_weights_only, desc_weights_only, model_weights=weights_only_data)

        ret_model_weights_only = storage.get_model_by_uuid(uuid_weights_only)
        self.assertIsNotNone(ret_model_weights_only)
        self.assertEqual(ret_model_weights_only['metadata']['user_name'], model_name_weights_only)
        np.testing.assert_array_equal(ret_model_weights_only['weights'], weights_only_data)
        self.assertIsNone(ret_model_weights_only['code'])


    def test_08_get_model_not_found(self):
        ret_model_uuid = storage.get_model_by_uuid("non_existent_model_uuid")
        self.assertIsNone(ret_model_uuid)
        ret_model_name = storage.get_model_by_name("non_existent_model_name")
        self.assertIsNone(ret_model_name)

    def test_09_save_model_name_overwrite(self):
        name = "overwrite_model"
        code1 = "print('v1')"
        weights1 = np.array([1,2])
        code2 = "print('v2')"
        weights2 = np.array([3,4])

        uuid1 = storage.save_model(name, "version1", model_weights=weights1, model_code=code1)
        ret_model1_name = storage.get_model_by_name(name)
        self.assertEqual(ret_model1_name['metadata']['description'], "version1")
        self.assertEqual(ret_model1_name['code'], code1)
        np.testing.assert_array_equal(ret_model1_name['weights'], weights1)

        uuid2 = storage.save_model(name, "version2", model_weights=weights2, model_code=code2)
        self.assertNotEqual(uuid1, uuid2)

        ret_model2_name = storage.get_model_by_name(name)
        self.assertEqual(ret_model2_name['metadata']['description'], "version2")
        self.assertEqual(ret_model2_name['code'], code2)
        np.testing.assert_array_equal(ret_model2_name['weights'], weights2)
        self.assertEqual(ret_model2_name['metadata']['uuid'], uuid2)

        # Check original UUID still fetches original model data
        ret_model_uuid1 = storage.get_model_by_uuid(uuid1)
        self.assertEqual(ret_model_uuid1['code'], code1)
        np.testing.assert_array_equal(ret_model_uuid1['weights'], weights1)

    def test_10_empty_tensor_save_and_get(self):
        tensor_name = "empty_tensor"
        description = "An empty tensor"
        data = np.array([]) # Empty numpy array

        uuid_val = storage.save_tensor(name=tensor_name, description=description, tensor_data=data)
        self.assertIsInstance(uuid_val, str)

        ret_data_uuid, ret_meta_uuid = storage.get_tensor_by_uuid(uuid_val)
        self.assertIsNotNone(ret_data_uuid)
        np.testing.assert_array_equal(ret_data_uuid, data) # Should be an empty array
        self.assertEqual(ret_meta_uuid['original_shape'], str(data.shape))
        self.assertEqual(ret_meta_uuid['original_dtype'], str(data.dtype))

    # --- Tests for _get_uuid_from_name (tested directly for simplicity) ---
    def test_11_get_uuid_from_name_helper(self):
        tensor_name1 = "helper_tensor_1"
        tensor_uuid1 = storage.save_tensor(tensor_name1, "desc", np.array([1]))
        model_name1 = "helper_model_1"
        model_uuid1 = storage.save_model(model_name1, "desc", model_code="pass")

        with h5py.File(TEST_HDF5_FILE, 'r') as hf:
            # Test found cases
            self.assertEqual(storage._get_uuid_from_name(hf, "tensor", tensor_name1), tensor_uuid1)
            self.assertEqual(storage._get_uuid_from_name(hf, "model", model_name1), model_uuid1)

            # Test not found cases
            self.assertIsNone(storage._get_uuid_from_name(hf, "tensor", "non_existent_tensor"))
            self.assertIsNone(storage._get_uuid_from_name(hf, "model", "non_existent_model"))

            # Test invalid item_type
            self.assertIsNone(storage._get_uuid_from_name(hf, "invalid_type", tensor_name1))

            # Test index dataset not present (harder to test without manipulating file structure directly)
            # This would typically be caught by the 'mapping_path not in hf' check if index was missing.

    # --- Tests for delete functions ---
    def test_12_delete_tensor_by_uuid(self):
        t1_name = "tensor_to_delete_1"
        t2_name = "tensor_to_keep_1"
        t1_uuid = storage.save_tensor(t1_name, "desc1", np.array([1,2]))
        t2_uuid = storage.save_tensor(t2_name, "desc2", np.array([3,4]))

        # Delete t1
        self.assertTrue(storage.delete_tensor_by_uuid(t1_uuid))

        # Verify t1 is deleted
        self.assertIsNone(storage.get_tensor_by_uuid(t1_uuid)[0])
        self.assertIsNone(storage.get_tensor_by_name(t1_name)[0]) # Check index removal

        # Verify t2 is still present
        t2_data, _ = storage.get_tensor_by_uuid(t2_uuid)
        self.assertIsNotNone(t2_data)
        np.testing.assert_array_equal(t2_data, np.array([3,4]))
        t2_data_name, _ = storage.get_tensor_by_name(t2_name)
        self.assertIsNotNone(t2_data_name)

        # Attempt to delete non-existent UUID
        self.assertFalse(storage.delete_tensor_by_uuid("non-existent-uuid-123"))
        # Attempt to delete already deleted UUID
        self.assertFalse(storage.delete_tensor_by_uuid(t1_uuid))


    def test_13_delete_model_by_uuid(self):
        m1_name = "model_to_delete_1"
        m2_name = "model_to_keep_1"
        m1_uuid = storage.save_model(m1_name, "desc1_model", model_code="print('m1')")
        m2_uuid = storage.save_model(m2_name, "desc2_model", model_code="print('m2')")

        self.assertTrue(storage.delete_model_by_uuid(m1_uuid))

        self.assertIsNone(storage.get_model_by_uuid(m1_uuid))
        self.assertIsNone(storage.get_model_by_name(m1_name)) # Check index removal

        m2_data = storage.get_model_by_uuid(m2_uuid)
        self.assertIsNotNone(m2_data)
        self.assertEqual(m2_data['metadata']['user_name'], m2_name)
        m2_data_name = storage.get_model_by_name(m2_name)
        self.assertIsNotNone(m2_data_name)

        self.assertFalse(storage.delete_model_by_uuid("non-existent-uuid-456"))
        self.assertFalse(storage.delete_model_by_uuid(m1_uuid))

    # --- Tests for list functions ---
    def test_14_list_tensors(self):
        # Empty state
        results, total = storage.list_tensors(None, 10, 0)
        self.assertEqual(len(results), 0)
        self.assertEqual(total, 0)

        # Add items (names chosen for sorting tests later)
        uuid1 = storage.save_tensor("apple_tensor", "desc a", np.array([1]))
        uuid2 = storage.save_tensor("Banana_tensor", "desc B", np.array([2])) # Note: case for sorting
        uuid3 = storage.save_tensor("cherry_tensor", "desc c", np.array([3]))
        uuid4 = storage.save_tensor("another_apple", "desc aa", np.array([4]))
        uuid5 = storage.save_tensor("TEST_APPLE", "desc TA", np.array([5]))


        # Basic listing & pagination
        results, total = storage.list_tensors(None, limit=2, offset=0)
        self.assertEqual(len(results), 2)
        self.assertEqual(total, 5)
        self.assertEqual(results[0]['user_name'], "another_apple") # Sorted by name: another_apple, apple_tensor
        self.assertEqual(results[1]['user_name'], "apple_tensor")

        results, total = storage.list_tensors(None, limit=2, offset=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(total, 5)
        self.assertEqual(results[0]['user_name'], "Banana_tensor") # banana_tensor, cherry_tensor
        self.assertEqual(results[1]['user_name'], "cherry_tensor")

        results, total = storage.list_tensors(None, limit=2, offset=4)
        self.assertEqual(len(results), 1)
        self.assertEqual(total, 5)
        self.assertEqual(results[0]['user_name'], "TEST_APPLE") # TEST_APPLE

        # Filtering
        results, total = storage.list_tensors(filter_by_name_contains="apple", limit=10, offset=0)
        self.assertEqual(len(results), 3) # another_apple, apple_tensor, TEST_APPLE
        self.assertEqual(total, 3)
        found_names = sorted([r['user_name'] for r in results])
        self.assertListEqual(found_names, sorted(["another_apple", "apple_tensor", "TEST_APPLE"]))

        results, total = storage.list_tensors(filter_by_name_contains="nonexistent", limit=10, offset=0)
        self.assertEqual(len(results), 0)
        self.assertEqual(total, 0)

        # Metadata content
        results, _ = storage.list_tensors(filter_by_name_contains="apple_tensor", limit=1, offset=0)
        self.assertEqual(len(results), 1)
        meta = results[0]
        self.assertEqual(meta['uuid'], uuid1)
        self.assertEqual(meta['user_name'], "apple_tensor")
        self.assertEqual(meta['description'], "desc a")
        self.assertTrue('creation_date' in meta)
        self.assertTrue('original_dtype' in meta)
        self.assertTrue('original_shape' in meta)


    def test_15_list_models(self):
        # Empty state
        results, total = storage.list_models(None, 10, 0)
        self.assertEqual(len(results), 0)
        self.assertEqual(total, 0)

        # Add items
        uuid_m1 = storage.save_model("alpha_model_code", "desc alpha code", model_code="code1")
        uuid_m2 = storage.save_model("Beta_model_weights", "desc Beta weights", model_weights=np.array([1]))
        uuid_m3 = storage.save_model("gamma_model_both", "desc gamma both", model_code="code3", model_weights=np.array([2]))

        # Basic listing (check sorting and content)
        results, total = storage.list_models(None, limit=3, offset=0)
        self.assertEqual(len(results), 3)
        self.assertEqual(total, 3)
        self.assertEqual(results[0]['user_name'], "alpha_model_code")
        self.assertEqual(results[1]['user_name'], "Beta_model_weights")
        self.assertEqual(results[2]['user_name'], "gamma_model_both")

        # Check metadata for one model
        model_alpha_meta = results[0]
        self.assertEqual(model_alpha_meta['uuid'], uuid_m1)
        self.assertTrue(model_alpha_meta['has_code'])
        self.assertFalse(model_alpha_meta['has_weights'])
        self.assertTrue('upload_date' in model_alpha_meta)

        model_beta_meta = results[1]
        self.assertFalse(model_beta_meta['has_code'])
        self.assertTrue(model_beta_meta['has_weights'])

        model_gamma_meta = results[2]
        self.assertTrue(model_gamma_meta['has_code'])
        self.assertTrue(model_gamma_meta['has_weights'])

        # Filtering
        results, total = storage.list_models(filter_by_name_contains="Beta", limit=10, offset=0)
        self.assertEqual(len(results), 1)
        self.assertEqual(total, 1)
        self.assertEqual(results[0]['user_name'], "Beta_model_weights")

    # --- Tests for update metadata functions ---
    def test_16_update_tensor_metadata(self):
        tensor_uuid = storage.save_tensor("orig_tensor_name", "orig_desc", np.array([1,2]))

        # Update description only
        updated_meta = storage.update_tensor_metadata(tensor_uuid, {"description": "new_desc"})
        self.assertIsNotNone(updated_meta)
        self.assertEqual(updated_meta['description'], "new_desc")
        self.assertEqual(updated_meta['user_name'], "orig_tensor_name") # Name should be unchanged

        retrieved_data, retrieved_meta = storage.get_tensor_by_uuid(tensor_uuid)
        self.assertEqual(retrieved_meta['description'], "new_desc")

        # Update name
        updated_meta_name_change = storage.update_tensor_metadata(tensor_uuid, {"user_name": "new_tensor_name"})
        self.assertIsNotNone(updated_meta_name_change)
        self.assertEqual(updated_meta_name_change['user_name'], "new_tensor_name")

        # Verify with get_by_uuid and get_by_name
        _, meta_by_uuid = storage.get_tensor_by_uuid(tensor_uuid)
        self.assertEqual(meta_by_uuid['user_name'], "new_tensor_name")

        _, meta_by_new_name = storage.get_tensor_by_name("new_tensor_name")
        self.assertIsNotNone(meta_by_new_name)
        self.assertEqual(meta_by_new_name['uuid'], tensor_uuid)

        _, meta_by_old_name = storage.get_tensor_by_name("orig_tensor_name")
        self.assertIsNone(meta_by_old_name) # Old name should no longer resolve to this tensor if name changed

        # Update both
        storage.update_tensor_metadata(tensor_uuid, {"user_name": "final_name", "description": "final_desc"})
        _, meta_final = storage.get_tensor_by_uuid(tensor_uuid)
        self.assertEqual(meta_final['user_name'], "final_name")
        self.assertEqual(meta_final['description'], "final_desc")

        # Update non-existent UUID
        self.assertIsNone(storage.update_tensor_metadata("non-existent-uuid", {"description": "bla"}))

        # Update with non-allowed fields (should be ignored)
        storage.update_tensor_metadata(tensor_uuid, {"creation_date": "test_date_overwrite_attempt"})
        _, meta_after_non_allowed = storage.get_tensor_by_uuid(tensor_uuid)
        self.assertNotEqual(meta_after_non_allowed['creation_date'], "test_date_overwrite_attempt")


    def test_17_update_model_metadata(self):
        model_uuid = storage.save_model("orig_model_name", "orig_model_desc", model_code="pass")

        # Update description
        updated_meta = storage.update_model_metadata(model_uuid, {"description": "new_model_desc"})
        self.assertIsNotNone(updated_meta)
        self.assertEqual(updated_meta['description'], "new_model_desc")
        self.assertEqual(updated_meta['user_name'], "orig_model_name")
        self.assertTrue(updated_meta['has_code']) # Check other fields are preserved

        # Update name
        storage.update_model_metadata(model_uuid, {"user_name": "new_model_name"})
        _, meta_by_uuid = storage.get_model_by_uuid(model_uuid)
        self.assertEqual(meta_by_uuid['user_name'], "new_model_name")

        _, meta_by_new_name = storage.get_model_by_name("new_model_name")
        self.assertIsNotNone(meta_by_new_name)
        self.assertEqual(meta_by_new_name['uuid'], model_uuid)

        _, meta_by_old_name = storage.get_model_by_name("orig_model_name")
        self.assertIsNone(meta_by_old_name)

        # Update non-existent UUID
        self.assertIsNone(storage.update_model_metadata("non-existent-uuid", {"description": "bla"}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
