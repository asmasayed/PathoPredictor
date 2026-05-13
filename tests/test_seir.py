"""
Tests for SEIR model simulation.
"""

import unittest
import numpy as np
from src.simulation.seir_model import simulate_seir

class TestSEIR(unittest.TestCase):
    """Test cases for SEIR model."""
    
    def test_seir_simulation(self):
        """Test basic SEIR simulation."""
        initial_conditions = [999, 1, 0, 0]  # S, E, I, R
        t = np.linspace(0, 100, 1000)
        N = 1000
        beta, gamma, sigma = 0.3, 0.1, 0.2
        
        solution = simulate_seir(initial_conditions, t, beta, gamma, sigma, N)
        
        # Check that solution has correct shape
        self.assertEqual(solution.shape, (len(t), 4))
        
        # Check that total population is conserved
        total = solution.sum(axis=1)
        np.testing.assert_allclose(total, N, rtol=1e-10)

if __name__ == '__main__':
    unittest.main()
