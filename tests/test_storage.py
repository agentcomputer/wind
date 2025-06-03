# tests/test_storage.py
import unittest
import os
import numpy as np
import h5py
from datetime import datetime

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
