
import unittest
import numpy as np
from surface_shock import SurfaceShockModel

class TestSurfaceShockModel(unittest.TestCase):
    def test_initialization(self):
        model = SurfaceShockModel(latent_dim=4)
        self.assertEqual(model.latent_dim, 4)
        self.assertFalse(model._is_trained)

    def test_train_stub(self):
        model = SurfaceShockModel()
        # Mock surface data (T, K, T_exp)
        dummy_data = np.zeros((10, 5, 3))
        model.train(dummy_data)
        self.assertTrue(model._is_trained)

    def test_sample_shape(self):
        model = SurfaceShockModel()
        n_scenarios = 100
        shocks = model.sample(current_surface=None, n_scenarios=n_scenarios, seed=42)
        
        # In the skeleton, we expect a flat array of shocks for now
        self.assertEqual(len(shocks), n_scenarios)
        self.assertIsInstance(shocks, np.ndarray)

    def test_reproducibility(self):
        model = SurfaceShockModel()
        shocks1 = model.sample(None, n_scenarios=10, seed=123)
        shocks2 = model.sample(None, n_scenarios=10, seed=123)
        np.testing.assert_array_equal(shocks1, shocks2)
        
        shocks3 = model.sample(None, n_scenarios=10, seed=999)
        self.assertFalse(np.array_equal(shocks1, shocks3))

if __name__ == '__main__':
    unittest.main()
