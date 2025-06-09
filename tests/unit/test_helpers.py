import unittest

from darcyflow.helpers import gidx


class TestGidx(unittest.TestCase):

    def test_zero_index(self):
        self.assertEqual(gidx(0, 0, 10), 0)

    def test_basic_indices(self):
        self.assertEqual(gidx(1, 0, 10), 1)
        self.assertEqual(gidx(0, 1, 10), 10)
        self.assertEqual(gidx(5, 2, 10), 25)

    def test_large_Nx(self):
        self.assertEqual(gidx(2, 3, 100), 302)
        self.assertEqual(gidx(99, 0, 100), 99)
        self.assertEqual(gidx(0, 99, 100), 9900)

    def test_negative_indices(self):
        with self.assertRaises(ValueError):
            gidx(-1, 0, 10)
        with self.assertRaises(ValueError):
            gidx(0, -1, 10)

    def test_non_integer_input(self):
        with self.assertRaises(TypeError):
            gidx(1.5, 2, 10)
        with self.assertRaises(TypeError):
            gidx(1, '2', 10)

if __name__ == '__main__':
    unittest.main()