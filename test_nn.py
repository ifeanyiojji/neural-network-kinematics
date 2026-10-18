import unittest

import numpy as np

from nn import MLP


class TestMLP(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        rng = np.random.default_rng(17)
        inputs = rng.normal(size=(8, 2))
        targets = rng.normal(size=(8, 2))
        network = MLP([2, 5, 4, 2], seed=3)

        error = network.gradient_check(inputs, targets)

        self.assertLess(error, 1e-6)

    def test_jacobian_matches_finite_differences(self):
        network = MLP([2, 6, 5, 2], seed=11)
        point = np.array([0.37, -0.61])
        step = 1e-6
        numerical = np.column_stack([
            (network.forward((point + step * np.eye(2)[index])[None, :])[0]
             - network.forward((point - step * np.eye(2)[index])[None, :])[0])
            / (2 * step)
            for index in range(2)
        ])

        np.testing.assert_allclose(network.jacobian(point), numerical, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()