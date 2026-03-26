import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "util.py"


def load_util_module():
    spec = importlib.util.spec_from_file_location("cloud_util_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CloudUtilTests(unittest.TestCase):
    def test_max_fn_clips_negative_mass_and_normalizes(self):
        util = load_util_module()

        probs = util.max_fn(np.array([-0.5, 0.2, 0.3]))

        np.testing.assert_allclose(probs, np.array([0.0, 0.4, 0.6]))

    def test_sample_returns_index_from_distribution(self):
        util = load_util_module()

        token = util.sample(np.array([0.0, 1.0, 0.0]), seed=1234)

        self.assertEqual(token, 1)


if __name__ == "__main__":
    unittest.main()
