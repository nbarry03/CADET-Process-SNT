"""
Todo
----
Add tests for
- section dependent parameters, polynomial parameters
"""
import unittest

import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import (
    Inlet, Cstr,
    TubularReactor, LumpedRateModelWithPores, LumpedRateModelWithoutPores, MCT
)

length = 0.6
diameter = 0.024

cross_section_area = np.pi/4 * diameter**2
volume_liquid = cross_section_area * length
volume = cross_section_area * length

bed_porosity = 0.3
particle_porosity = 0.6
particle_radius = [1e-4]
par_type_volfrac = [1]
total_porosity = bed_porosity + (1 - bed_porosity) * particle_porosity
const_solid_volume = volume * (1 - total_porosity)
init_liquid_volume = volume * total_porosity

axial_dispersion = 4.7e-7

channel_cross_section_areas = [0.1,0.1,0.1]
exchange_matrix = np.array([
                             [[0.0],[0.01],[0.0]],
                             [[0.02],[0.0],[0.03]],
                             [[0.0],[0.0],[0.0]]
                             ])
flow_direction = 1


class Test_Unit_Operation(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def setUp(self):
        self.component_system = ComponentSystem(2)

    def create_source(self):
        source = Inlet(self.component_system, name='test')

        return source

    def create_cstr(self):
        cstr = Cstr(self.component_system, name='test')

        cstr.const_solid_volume = const_solid_volume
        cstr.init_liquid_volume = init_liquid_volume

        cstr.flow_rate = 1

        return cstr

    def create_tubular_reactor(self):
        tube = TubularReactor(self.component_system, name='test')

@pytest.fixture
def lrmp(component_system):
    lrmp = LumpedRateModelWithPores(component_system, name="test_lrmp")
    lrmp.length = length
    lrmp.diameter = diameter
    lrmp.axial_dispersion = axial_dispersion
    lrmp.bed_porosity = bed_porosity
    lrmp.particle_radius = particle_radius
    lrmp.par_type_volfrac = par_type_volfrac
    lrmp.particle_porosity = particle_porosity
    lrmp.film_diffusion = [film_diffusion_0, film_diffusion_1]
    return lrmp

        return tube

@pytest.fixture
def grm(components=2):
    grm = GeneralRateModel(ComponentSystem(components), name="test_grm")
    grm.length = length
    grm.diameter = diameter
    grm.axial_dispersion = axial_dispersion
    grm.bed_porosity = bed_porosity
    grm.particle_radius = particle_radius
    grm.par_type_volfrac = par_type_volfrac
    grm.particle_porosity = particle_porosity
    grm.film_diffusion = [film_diffusion_0, film_diffusion_1]
    grm.pore_diffusion = [pore_diffusion_0, pore_diffusion_1]
    grm.discretization.npar = [5]
    return grm

        mct.length = length
        mct.channel_cross_section_areas = channel_cross_section_areas
        mct.axial_dispersion = 0
        mct.flow_direction = flow_direction

        return mct

    def create_lrmwop(self):
        lrmwop = LumpedRateModelWithoutPores(
            self.component_system, name='test'
        )

        lrmwop.length = length
        lrmwop.diameter = diameter
        lrmwop.axial_dispersion = axial_dispersion
        lrmwop.total_porosity = total_porosity

        return lrmwop

    if "cross_section_area" in expected_geometry:
        assert unit.cross_section_area == expected_geometry["cross_section_area"]

        unit.cross_section_area = cross_section_area / 2
        assert np.isclose(unit.diameter, diameter / (2**0.5))


@pytest.mark.parametrize(
    "input_c, expected_c",
    [
        (1, np.array([[1, 0, 0, 0], [1, 0, 0, 0]])),
        ([1, 1], np.array([[1, 0, 0, 0], [1, 0, 0, 0]])),
        ([1, 2], np.array([[1, 0, 0, 0], [2, 0, 0, 0]])),
        ([[1, 0], [2, 0]], np.array([[1, 0, 0, 0], [2, 0, 0, 0]])),
        ([[1, 2], [3, 4]], np.array([[1, 2, 0, 0], [3, 4, 0, 0]])),
    ],
)
def test_polynomial_inlet_concentration(inlet, input_c, expected_c):
    inlet.c = input_c
    np.testing.assert_equal(inlet.c, expected_c)


@pytest.mark.parametrize(
    "unit_operation, input_flow_rate, expected_flow_rate",
    [
        ("inlet", 1, np.array([1, 0, 0, 0])),
        ("inlet", [1, 0], np.array([1, 0, 0, 0])),
        ("inlet", [1, 1], np.array([1, 1, 0, 0])),
        ("cstr", 1, np.array([1, 0, 0, 0])),
        ("cstr", [1, 0], np.array([1, 0, 0, 0])),
        ("cstr", [1, 1], np.array([1, 1, 0, 0])),
    ],
)
def test_polynomial_flow_rate(
    unit_operation, input_flow_rate, expected_flow_rate, request
):
    unit = request.getfixturevalue(unit_operation)
    unit.flow_rate = input_flow_rate
    np.testing.assert_equal(unit.flow_rate, expected_flow_rate)


@pytest.mark.parametrize(
    "unit_operation, expected_parameters",
    [
        (
            "cstr",
            {
                "flow_rate": np.array([1, 0, 0, 0]),
                "init_liquid_volume": init_liquid_volume,
                "flow_rate_filter": 0,
                "c": [0, 0],
                "q": None,
                "const_solid_volume": const_solid_volume,
            },
        ),
        (
            "tubular_reactor",
            {
                "length": length,
                "diameter": diameter,
                "axial_dispersion": [axial_dispersion, axial_dispersion],
                "flow_direction": flow_direction,
                "c": [0, 0],
                "discretization": {
                    "ncol": 100,
                    "use_analytic_jacobian": True,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": "LEVMAR",
                    },
                },
            },
        ),
        (
            "lrm",
            {
                "length": length,
                "diameter": diameter,
                "total_porosity": total_porosity,
                "axial_dispersion": [axial_dispersion, axial_dispersion],
                "flow_direction": flow_direction,
                "c": [0, 0],
                "q": None,
                "discretization": {
                    "ncol": 100,
                    "use_analytic_jacobian": True,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": "LEVMAR",
                    },
                },
            },
        ),
        (
            "lrmp",
            {
                "length": length,
                "diameter": diameter,
                "bed_porosity": bed_porosity,
                "axial_dispersion": [axial_dispersion, axial_dispersion],
                "pore_accessibility": [1, 1],
                "film_diffusion": [film_diffusion_0, film_diffusion_1],
                "particle_radius": particle_radius,
                "par_type_volfrac": par_type_volfrac,
                "particle_porosity": particle_porosity,
                "flow_direction": flow_direction,
                "c": [0, 0],
                "cp": [0, 0],
                "q": None,
                "discretization": {
                    "ncol": 100,
                    "par_geom": "SPHERE",
                    "use_analytic_jacobian": True,
                    "gs_type": True,
                    "max_krylov": 0,
                    "max_restarts": 10,
                    "schur_safety": 1e-08,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": "LEVMAR",
                    },
                },
            },
        ),
        (
            "grm",
            {
                "length": length,
                "diameter": diameter,
                "bed_porosity": bed_porosity,
                "axial_dispersion": [axial_dispersion, axial_dispersion],
                "pore_accessibility": [1, 1],
                "film_diffusion": [film_diffusion_0, film_diffusion_1],
                "particle_radius": particle_radius,
                "par_type_volfrac": par_type_volfrac,
                "particle_porosity": particle_porosity,
                "pore_diffusion": [pore_diffusion_0, pore_diffusion_1],
                "surface_diffusion": None,
                "flow_direction": flow_direction,
                "c": [0, 0],
                "cp": [0, 0],
                "q": None,
                "discretization": {
                    "ncol": 100,
                    "par_geom": "SPHERE",
                    "npar": [5],
                    "par_disc_type": "EQUIDISTANT_PAR",
                    "par_boundary_order": 2,
                    "fix_zero_surface_diffusion": False,
                    "use_analytic_jacobian": True,
                    "gs_type": True,
                    "max_krylov": 0,
                    "max_restarts": 10,
                    "schur_safety": 1e-08,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": "LEVMAR",
                    },
                },
            },
        ),
        (
            "mct",
            {
                "nchannel": nchannel,
                "length": length,
                "channel_cross_section_areas": channel_cross_section_areas,
                "axial_dispersion": nchannel * [axial_dispersion],
                "exchange_matrix": exchange_matrix,
                "flow_direction": 1,
                "c": [[0, 0, 0]],
                "discretization": {
                    "ncol": 100,
                    "use_analytic_jacobian": True,
                    "reconstruction": "WENO",
                    "weno": {"boundary_model": 0, "weno_eps": 1e-10, "weno_order": 3},
                    "consistency_solver": {
                        "solver_name": "LEVMAR",
                        "init_damping": 0.01,
                        "min_damping": 0.0001,
                        "max_iterations": 50,
                        "subsolvers": "LEVMAR",
                    },
                },
            },
        ),
    ],
)
def test_parameters(unit_operation, expected_parameters, request):
    unit = request.getfixturevalue(unit_operation)
    np.testing.assert_equal(expected_parameters, unit.parameters)


@pytest.mark.parametrize(
    "unit_operation, flow_rate, expected_velocity",
    [
        ("tubular_reactor", 2, 2),
        ("lrmp", 2, 4),
        ("tubular_reactor", 0, ZeroDivisionError),
        ("lrmp", 0, ZeroDivisionError),
    ],
)
def test_interstitial_velocity(unit_operation, flow_rate, expected_velocity, request):
    unit = request.getfixturevalue(unit_operation)
    unit.length = 1
    unit.cross_section_area = 1
    unit.axial_dispersion = 3 if unit_operation == "tubular_reactor" else [3, 2]

    if unit_operation == "lrmp":
        unit.bed_porosity = 0.5

    if expected_velocity is ZeroDivisionError:
        with pytest.raises(ZeroDivisionError):
            unit.calculate_interstitial_velocity(flow_rate)
    else:
        assert np.isclose(
            unit.calculate_interstitial_velocity(flow_rate), expected_velocity
        )

        lrmwp.length = length
        lrmwp.diameter = diameter
        lrmwp.axial_dispersion = axial_dispersion
        lrmwp.bed_porosity = bed_porosity
        lrmwp.particle_porosity = particle_porosity

        return lrmwp

    def test_geometry(self):
        cstr = self.create_cstr()
        lrmwop = self.create_lrmwop()
        lrmwp = self.create_lrmwp()

        self.assertEqual(lrmwop.cross_section_area, cross_section_area)
        self.assertEqual(lrmwp.cross_section_area, cross_section_area)

        self.assertEqual(lrmwop.total_porosity, total_porosity)
        self.assertEqual(lrmwp.total_porosity, total_porosity)

        self.assertEqual(lrmwop.volume, volume)
        self.assertEqual(lrmwp.volume, volume)

        volume_interstitial = total_porosity * volume
        self.assertAlmostEqual(lrmwop.volume_interstitial, volume_interstitial)
        volume_interstitial = bed_porosity * volume
        self.assertAlmostEqual(lrmwp.volume_interstitial, volume_interstitial)

        volume_liquid = total_porosity * volume
        self.assertAlmostEqual(cstr.volume_liquid, volume_liquid)
        self.assertAlmostEqual(lrmwop.volume_liquid, volume_liquid)
        self.assertAlmostEqual(lrmwp.volume_liquid, volume_liquid)

        volume_solid = (1 - total_porosity) * volume
        self.assertAlmostEqual(cstr.volume_solid, volume_solid)
        self.assertAlmostEqual(lrmwop.volume_solid, volume_solid)
        self.assertAlmostEqual(lrmwp.volume_solid, volume_solid)

        lrmwop.cross_section_area = cross_section_area/2
        self.assertAlmostEqual(lrmwop.diameter, diameter/(2**0.5))

    def test_convection_dispersion(self):
        tube = self.create_tubular_reactor()
        lrmwp = self.create_lrmwp()

        flow_rate = 0
        tube.length = 1
        tube.cross_section_area = 1
        tube.axial_dispersion = 0

        with self.assertRaises(ZeroDivisionError):
            tube.calculate_interstitial_velocity(flow_rate)
        with self.assertRaises(ZeroDivisionError):
            tube.calculate_superficial_velocity(flow_rate)
        with self.assertRaises(ZeroDivisionError):
            tube.NTP(flow_rate)

        flow_rate = 2
        tube.axial_dispersion = 3
        self.assertAlmostEqual(tube.calculate_interstitial_velocity(flow_rate), 2)
        self.assertAlmostEqual(tube.calculate_interstitial_rt(flow_rate), 0.5)
        self.assertAlmostEqual(tube.calculate_superficial_velocity(flow_rate), 2)
        self.assertAlmostEqual(tube.calculate_superficial_rt(flow_rate), 0.5)
        self.assertAlmostEqual(tube.NTP(flow_rate), 1/3)

        tube.set_axial_dispersion_from_NTP(1/3, 2)
        self.assertAlmostEqual(tube.axial_dispersion, 3)

        flow_rate = 2
        lrmwp.length = 1
        lrmwp.bed_porosity = 0.5
        lrmwp.cross_section_area = 1
        self.assertAlmostEqual(lrmwp.calculate_interstitial_velocity(flow_rate), 4)
        self.assertAlmostEqual(lrmwp.calculate_interstitial_rt(flow_rate), 0.25)
        self.assertAlmostEqual(lrmwp.calculate_superficial_velocity(flow_rate), 2)
        self.assertAlmostEqual(lrmwp.calculate_superficial_rt(flow_rate), 0.5)

    def test_poly_properties(self):
        source = self.create_source()

        ref = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
        source.c = 1
        np.testing.assert_equal(source.c, ref)
        source.c = [1, 1]
        np.testing.assert_equal(source.c, ref)

        ref = np.array([[1, 0, 0, 0], [2, 0, 0, 0]])
        source.c = [1, 2]
        np.testing.assert_equal(source.c, ref)
        source.c = [[1, 0], [2, 0]]
        np.testing.assert_equal(source.c, ref)

        ref = np.array([[1, 2, 0, 0], [3, 4, 0, 0]])
        source.c = [[1, 2], [3, 4]]
        np.testing.assert_equal(source.c, ref)
        source.c = ref
        np.testing.assert_equal(source.c, ref)

        cstr = self.create_cstr()

        ref = np.array([1, 0, 0, 0])
        cstr.flow_rate = 1
        np.testing.assert_equal(cstr.flow_rate, ref)
        cstr.flow_rate = [1, 0]
        np.testing.assert_equal(cstr.flow_rate, ref)

        ref = np.array([1, 1, 0, 0])
        cstr.flow_rate = [1, 1]
        np.testing.assert_equal(cstr.flow_rate, ref)
        cstr.flow_rate = ref
        np.testing.assert_equal(cstr.flow_rate, ref)

    def test_parameters(self):
        """
        Notes
        -----
            Currently, only getting parameters is tested. Should also test if
            setting works. For this, adsorption parameters should be provided.
        """
        cstr = self.create_cstr()
        parameters_expected = {
                'flow_rate': np.array([1, 0, 0, 0]),
                'init_liquid_volume': init_liquid_volume,
                'flow_rate_filter': 0,
                'c': [0, 0],
                'q': [],
                'const_solid_volume': const_solid_volume,
        }

        np.testing.assert_equal(parameters_expected, cstr.parameters)

        sec_dep_parameters_expected = {
                'flow_rate': np.array([1, 0, 0, 0]),
                'flow_rate_filter': 0,
        }
        np.testing.assert_equal(
            sec_dep_parameters_expected, cstr.section_dependent_parameters
        )

        poly_parameters = {
            'flow_rate': np.array([1, 0, 0, 0]),
        }
        np.testing.assert_equal(
            poly_parameters, cstr.polynomial_parameters
        )

        self.assertEqual(cstr.required_parameters, ['init_liquid_volume'])


    def test_MCT(self):
        """
        Notes
        -----
            Tests Parameters, Volumes and Attributes depending on nchannel. Should be later integrated into general testing workflow.
        """
        total_porosity = 1

        mct = self.create_MCT(1)

        mct.exchange_matrix = exchange_matrix

        parameters_expected = {
        'c': np.array([[0., 0., 0.]]),
        'axial_dispersion' : 0,
        'channel_cross_section_areas' : channel_cross_section_areas,
        'length' : length,
        'exchange_matrix': exchange_matrix,
        'flow_direction' : 1,
        'nchannel' : 3
        }
        np.testing.assert_equal(parameters_expected, {key: value for key, value in mct.parameters.items() if key != 'discretization'})

        volume = length*sum(channel_cross_section_areas)
        volume_liquid = volume*total_porosity
        volume_solid = (total_porosity-1)*volume

        self.assertAlmostEqual(mct.volume_liquid, volume_liquid)
        self.assertAlmostEqual(mct.volume_solid, volume_solid)

        with self.assertRaises(ValueError):
            mct.exchange_matrix =  np.array([[
                             [0.0, 0.01, 0.0],
                             [0.02, 0.0, 0.03],
                             [0.0, 0.0, 0.0]
                             ]])

        mct.nchannel = 2
        with self.assertRaises(ValueError):
            mct.exchange_matrix
            mct.channel_cross_section_areas

        self.assertTrue(mct.nchannel*mct.component_system.n_comp == mct.c.size)

        mct2 = self.create_MCT(2)

        with self.assertRaises(ValueError):
            mct2.exchange_matrix =  np.array([[
                            [0.0, 0.01, 0.0],
                            [0.02, 0.0, 0.03],
                            [0.0, 0.0, 0.0]
                            ],

                            [
                            [0.0, 0.01, 0.0],
                            [0.02, 0.0, 0.03],
                            [0.0, 0.0, 0.0]
                            ]])


if __name__ == '__main__':
    unittest.main()
