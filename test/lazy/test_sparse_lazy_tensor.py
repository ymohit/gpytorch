import unittest

import torch

from gpytorch.lazy.sparse_lazy_tensor import SparseLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestSparseLazyTensor(LazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        i = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 0, 1, 2, 3]])
        v = torch.tensor([3.0, 1.5, 1.5, 2.0, 5.0, 6.0], requires_grad=True)
        return SparseLazyTensor(indices=i, values=v, sparse_size=torch.Size([4, 4]))

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.to_dense()


if __name__ == "__main__":
    unittest.main()
