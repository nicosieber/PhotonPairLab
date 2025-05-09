import unittest
import random
from photonpairlab.crystal import Crystal
from photonpairlab.materials import KTP1, KTP2, KTP3

class TestCrystal(unittest.TestCase):
    def setUp(self):
        # Define default configuration
        self.default_config = {
            "Lc": 45e-6,
            "Lo": 30e-3,
            "T": 25,
            "w": 20e-6,
            "material": random.choice([KTP1(), KTP2(), KTP3()]),
            "spdc": random.choice(["type-II"])
        }
        # Initialize Crystal with default configuration
        self.crystal = Crystal(**self.default_config)

    def test_initialization(self):
        # Check that basic attributes are correctly set
        self.assertIsInstance(self.crystal.Lc, (int, float))
        self.assertIsInstance(self.crystal.Lo, (int, float))
        self.assertIsInstance(self.crystal.T, (int, float))
        self.assertIsInstance(self.crystal.w, (int, float))
        self.assertTrue(isinstance(self.crystal.material, (KTP1, KTP2, KTP3)))
        self.assertIn(self.crystal.spdc, ["type-0", "type-I", "type-II"])

    def test_periodic_poling(self):
        self.crystal.generate_periodic_poling(resolution=5)
        self.assertIsNotNone(self.crystal.sarray) # Check if crystal poling array is generated
        self.assertEqual(len(self.crystal.sarray), len(self.crystal.z)) # Check if the length of sarray matches the length of z
        self.assertTrue(all(x in [-1, 1] for x in self.crystal.sarray)) # Check if poling direction is either -1 or 1

if __name__ == '__main__':
    unittest.main()
