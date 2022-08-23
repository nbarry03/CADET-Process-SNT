import unittest

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.optimization import Individual, Population, ParetoFront

enable_plot = False


class TestPopulation(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        x = [1, 2]
        f = [-1]
        self.individual_1 = Individual(x, f)

        x = [2, 3]
        f = [-2]
        self.individual_2 = Individual(x, f)

        x = [1.001, 2]
        f = [-1.001]
        self.individual_similar = Individual(x, f)

        x = [1, 2]
        f = [-1, -2]
        self.individual_multi_1 = Individual(x, f)

        x = [1.001, 2]
        f = [-1.001, -2]
        self.individual_multi_2 = Individual(x, f)

        x = [1, 2]
        f = [-1]
        g = [3]
        self.individual_constr_1 = Individual(x, f, g)

        x = [2, 3]
        f = [-2]
        g = [0]
        self.individual_constr_2 = Individual(x, f, g)

        self.population = Population()
        self.population.add_individual(self.individual_1)
        self.population.add_individual(self.individual_2)
        self.population.add_individual(self.individual_similar)

        self.population_multi = Population()
        self.population_multi.add_individual(self.individual_multi_1)
        self.population_multi.add_individual(self.individual_multi_2)

        self.population_constr = Population()
        self.population_constr.add_individual(self.individual_constr_1)
        self.population_constr.add_individual(self.individual_constr_2)

    def test_values(self):
        x_expected = np.array([
            [1, 2],
            [2, 3],
            [1.001, 2]
        ])
        x = self.population.x
        np.testing.assert_almost_equal(x, x_expected)

        f_expected = np.array([
            [-1],
            [-2],
            [-1.001]
        ])
        f = self.population.f
        np.testing.assert_almost_equal(f, f_expected)

        g_expected = np.array([
            [3],
            [0]
        ])
        g = self.population_constr.g
        np.testing.assert_almost_equal(g, g_expected)

    def test_add_remove(self):
        with self.assertRaises(TypeError):
            self.population.add_individual('foo')

        self.population.add_individual(
            self.individual_1, ignore_duplicate=True
        )

        with self.assertRaises(CADETProcessError):
            self.population.add_individual(
                self.individual_1, ignore_duplicate=False
            )

        new_individual = Individual([9, 10], [3], [-1])

        new_individual = Individual([9, 10], [3], [-1])
        with self.assertRaises(CADETProcessError):
            self.population.add_individual(new_individual)

        self.population_constr.add_individual(new_individual)

        x_expected = np.array([
            [1, 2],
            [2, 3],
            [9, 10],
        ])
        x = self.population_constr.x
        np.testing.assert_almost_equal(x, x_expected)

        f_expected = np.array([
            [-1],
            [-2],
            [3]
        ])
        f = self.population_constr.f
        np.testing.assert_almost_equal(f, f_expected)

        g_expected = np.array([
            [3],
            [0],
            [-1]
        ])
        g = self.population_constr.g
        np.testing.assert_almost_equal(g, g_expected)

        with self.assertRaises(TypeError):
            self.population.remove_individual('foo')

        with self.assertRaises(CADETProcessError):
            self.population.remove_individual(new_individual)

        self.population_constr.remove_individual(new_individual)
        x_expected = np.array([
            [1, 2],
            [2, 3],
        ])
        x = self.population_constr.x
        np.testing.assert_almost_equal(x, x_expected)

        f_expected = np.array([
            [-1],
            [-2],
        ])
        f = self.population_constr.f
        np.testing.assert_almost_equal(f, f_expected)

        g_expected = np.array([
            [3],
            [0],
        ])
        g = self.population_constr.g
        np.testing.assert_almost_equal(g, g_expected)

    def test_min_max(self):
        f_min_expected = [-2]
        f_min = self.population_constr.f_min
        np.testing.assert_almost_equal(f_min, f_min_expected)

        f_max_expected = [-1]
        f_max = self.population_constr.f_max
        np.testing.assert_almost_equal(f_max, f_max_expected)

        g_min_expected = [0]
        g_min = self.population_constr.g_min
        np.testing.assert_almost_equal(g_min, g_min_expected)

        g_max_expected = [3]
        g_max = self.population_constr.g_max
        np.testing.assert_almost_equal(g_max, g_max_expected)

    def test_plot(self):
        if enable_plot:
            pass


class TestPareto(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        front = ParetoFront(3)

        population = [
            Individual([1, 2], [1, 2, 3]),
        ]

        front.update(population)

        population = [
            Individual([1, 2.01], [1, 2, 3.01]),
            Individual([4, 5], [4, 5, 6]),
        ]

        front.update(population)


if __name__ == '__main__':
    enable_plot = True

    unittest.main()