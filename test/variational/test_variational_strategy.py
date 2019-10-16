#!/usr/bin/env python3

import unittest
import gpytorch
import torch
from gpytorch.test.variational_test_case import VariationalTestCase


class TestVariationalGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return gpytorch.variational.VariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!


class TestPredictiveGP(TestVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveCrossEntropy


if __name__ == "__main__":
    unittest.main()
