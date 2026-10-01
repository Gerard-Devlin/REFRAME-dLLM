"""CPU flag lifetime tests; not evidence about GPU numeric errors or speed."""
import unittest

import torch

from focus_dllm.tuning.reduction_policy import BF16Reduction


class Checks(unittest.TestCase):
    def setUp(self):
        self.original=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction

    def tearDown(self):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=self.original

    def test_default_is_unchanged(self):
        with BF16Reduction() as policy:
            self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,self.original)
        self.assertTrue(policy.audit['restored'])

    def test_disable_and_restore(self):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        with BF16Reduction(True) as policy:
            self.assertFalse(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        self.assertTrue(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        self.assertTrue(policy.audit['restored'])

    def test_exception_restores(self):
        with self.assertRaises(RuntimeError):
            with BF16Reduction(True) as policy:raise RuntimeError('deliberate')
        self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,self.original)
        self.assertTrue(policy.audit['restored'])

    def test_nested_control_restores_outer_setting(self):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        with BF16Reduction(True):
            with BF16Reduction(True):pass
            self.assertFalse(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        self.assertTrue(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


if __name__=='__main__':unittest.main(verbosity=2)
