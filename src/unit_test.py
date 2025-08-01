import unittest
import numpy as np
from utils import retrieve_plus_minus_vectors_of_nearest_neighbour  # Replace with actual module name

# 🔧 Helper for debug-printing on assertion failures
def try_except(actual, expected, msg="", kind="equal"):
    try:
        if kind == "equal":
            np.testing.assert_array_equal(actual, expected)
        elif kind == "close":
            np.testing.assert_allclose(actual, expected)
        else:
            raise ValueError("Unknown kind. Use 'equal' or 'close'.")
    except AssertionError as e:
        print(f"\n[FAIL] {msg}:\nExpected:\n{expected}\nGot:\n{actual}")
        raise e

class test_retrieve_plus_minus_vectors_of_nearest_neighbour(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.N = 6
        self.D = 3
        self.diff_vectors = np.arange(self.N * self.N * self.D).reshape(self.N, self.N, self.D).astype(np.float32)
        self.nearest_indices = np.array([5, 3, 4, 3, 2, 0])  

    def test_shape_output(self):
        """✅ Shape test: Output should be (N, 3, D)"""
        minus, self_, plus = retrieve_plus_minus_vectors_of_nearest_neighbour(self.diff_vectors, self.nearest_indices)
        self.assertEqual(minus.shape, (self.N, 3, self.D))
        self.assertEqual(self_.shape, (self.N, 3, self.D))
        self.assertEqual(plus.shape, (self.N, 3, self.D))
        

    def test_masking_index_0(self):
        """✅ Masking at index 0: i-1 should fall back and be masked"""
        minus, _, _ = retrieve_plus_minus_vectors_of_nearest_neighbour(self.diff_vectors, self.nearest_indices)
        expected = self.diff_vectors[-1+1, 5-1] #i-1=-1 but -1+1=0 --> j-1=4
        actual = minus[0,0] #
        try_except(actual, expected, msg="test_masking_index_0")

    def test_masking_last_index(self):
        """✅ Masking at last index: i+1 should fall back and be masked"""
        _, _, plus = retrieve_plus_minus_vectors_of_nearest_neighbour(self.diff_vectors, self.nearest_indices)
        expected = self.diff_vectors[6-1, 0] #i+1=6 but 6-1=5 --> j=0
        actual = plus[6-1, 1]
        try_except(actual, expected, msg="test_masking_last_index")

    def test_correct_diff_vector_usage(self):
        """✅ Correct diff_vector usage: check known values manually"""
        minus, self_, plus = retrieve_plus_minus_vectors_of_nearest_neighbour(self.diff_vectors, self.nearest_indices)
        # as general rule, minus, self_ and plus are (N,3,D) dimensions
        # [N,0,D] --> j-1, [N,1,D] --> j and [N,2,D] --> j+1
        expected = self.diff_vectors[2, 4-1] # for i=2, nearest_j=4
        actual = self_[2, 0]
        try_except(actual, expected, msg="test_correct_diff_vector_usage (self_[2,0]), i-->j-1")

        expected = self.diff_vectors[1, 3] # for i=1, nearest_j=3
        actual = self_[1, 1]
        try_except(actual, expected, msg="test_correct_diff_vector_usage (self_[1,1]), i-->j")

        expected = self.diff_vectors[0, 5+1-1] # for i=0, nearest_j=5, and bcz it is the last, j+1=6 is out of range, so we put j+1-1=5
        actual = self_[0, 2]
        try_except(actual, expected, msg="test_correct_diff_vector_usage (self_[0,2]), i-->j+1")

        expected = self.diff_vectors[3-1, 3-1] # for i=3 (i-1=2) and nearest j=3 (j-1=2)
        actual = minus[3, 0]
        try_except(actual, expected, msg="test_correct_diff_vector_usage (minus[3,0]), i-1-->j-1")

        expected = self.diff_vectors[1+1, 3+1] # for i=1 (i+1=2) and nearest j=3 (j+1=4)
        actual = plus[1, 2]
        try_except(actual, expected, msg="test_correct_diff_vector_usage (plus[1,2]), i+1-->j+1")

    def test_random_but_reproducible(self):
        """✅ Random values are stable if seed is set"""
        np.random.seed(42)
        A = np.random.randn(3, 3, 3)
        expected = np.random.RandomState(42).randn(3, 3, 3)
        try_except(A, expected, msg="test_random_but_reproducible", kind="close")

if __name__ == '__main__':
    unittest.main()
