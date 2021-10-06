import unittest
import gravity
import numpy as np

class GravityTestCase(unittest.TestCase):
    def test_2(self):
        x = np.zeros(20)
        m = gravity.PoissonGravitationalModel()
        ydist = m(x)
        self.assertEqual(len(ydist.components),2)

    def test_3(self):
        x = np.zeros(20)
        m = gravity.PoissonGravitationalModel(nnz=2)
        ydist = m(x)
        self.assertEqual(len(ydist.components),3)

if __name__ == '__main__':
    unittest.main()
