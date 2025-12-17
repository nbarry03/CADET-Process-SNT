import unittest

import numpy as np
from CADETProcess.modelBuilder import (
    CarouselBuilder, SerialCarouselBuilder, ParallelZone, SerialZone
)
from CADETProcess.processModel import (
    ComponentSystem,
    Inlet,
    Linear,
    LumpedRateModelWithoutPores,
    Outlet,
)

from CADETProcess.simulator import Cadet


class Test_Carousel(unittest.TestCase):
    def setUp(self):
        self.component_system = ComponentSystem(2)

        self.binding_model = Linear(self.component_system)
        self.binding_model.adsorption_rate = [6, 8]
        self.binding_model.desorption_rate = [1, 1]

        self.column = [
            LumpedRateModelWithoutPores(self.component_system, name='upstream'),
            LumpedRateModelWithoutPores(self.component_system, name='downstream')
            ]
        for subunit in self.column:
            subunit.length = 0.6
            subunit.diameter = 0.024
            subunit.axial_dispersion = 4.7e-7
            subunit.total_porosity = 0.7

            subunit.binding_model = self.binding_model

    def create_serial(self):
        source = Inlet(self.component_system, name="source")
        source.c = [10, 10]
        source.flow_rate = 2e-7

        sink = Outlet(self.component_system, name="sink")

        serial_zone = SerialZone(self.component_system, "serial", 2, flow_direction=1)

        builder = CarouselBuilder(self.component_system, "serial")
        builder.column = self.column

        builder.add_unit(source)
        builder.add_unit(sink)
        builder.add_unit(serial_zone)

        builder.add_connection(source, serial_zone)
        builder.add_connection(serial_zone, sink)

        builder.switch_time = 300

        return builder

    def create_parallel(self):
        source = Inlet(self.component_system, name="source")
        source.c = [10, 10]
        source.flow_rate = 2e-7

        sink = Outlet(self.component_system, name="sink")

        parallel_zone = ParallelZone(
            self.component_system, "parallel", 2, flow_direction=1
        )

        builder = CarouselBuilder(self.component_system, "parallel")
        builder.column = self.column

        builder.add_unit(source)
        builder.add_unit(sink)
        builder.add_unit(parallel_zone)

        builder.add_connection(source, parallel_zone)
        builder.add_connection(parallel_zone, sink)

        builder.switch_time = 300

        return builder

    def create_smb(self):
        feed = Inlet(self.component_system, name="feed")
        feed.c = [10, 10]
        feed.flow_rate = 2e-7

        eluent = Inlet(self.component_system, name="eluent")
        eluent.c = [0, 0]
        eluent.flow_rate = 6e-7

        raffinate = Outlet(self.component_system, name="raffinate")
        extract = Outlet(self.component_system, name="extract")

        zone_I = SerialZone(self.component_system, "zone_I", 1)
        zone_II = SerialZone(self.component_system, "zone_II", 1)
        zone_III = SerialZone(self.component_system, "zone_III", 1)
        zone_IV = SerialZone(self.component_system, "zone_IV", 1)

        builder = CarouselBuilder(self.component_system, "smb")
        builder.column = self.column
        builder.add_unit(feed)
        builder.add_unit(eluent)

        builder.add_unit(raffinate)
        builder.add_unit(extract)

        builder.add_unit(zone_I)
        builder.add_unit(zone_II)
        builder.add_unit(zone_III)
        builder.add_unit(zone_IV)

        builder.add_connection(eluent, zone_I)

        builder.add_connection(zone_I, extract)
        builder.add_connection(zone_I, zone_II)
        w_e = 0.15
        builder.set_output_state(zone_I, [w_e, 1 - w_e])

        builder.add_connection(zone_II, zone_III)

        builder.add_connection(feed, zone_III)

        builder.add_connection(zone_III, raffinate)
        builder.add_connection(zone_III, zone_IV)
        w_r = 0.15
        builder.set_output_state(zone_III, [w_r, 1 - w_r])

        builder.add_connection(zone_IV, zone_I)

        builder.switch_time = 300

        return builder

    def create_multi_zone(self):
        source_serial = Inlet(self.component_system, name="source_serial")
        source_serial.c = [10, 10]
        source_serial.flow_rate = 2e-7

        sink_serial = Outlet(self.component_system, name="sink_serial")

        serial_zone = SerialZone(self.component_system, "serial", 2, flow_direction=1)

        source_parallel = Inlet(self.component_system, name="source_parallel")
        source_parallel.c = [10, 10]
        source_parallel.flow_rate = 2e-7

        sink_parallel = Outlet(self.component_system, name="sink_parallel")

        parallel_zone = ParallelZone(
            self.component_system, "parallel", 2, flow_direction=-1
        )

        builder = CarouselBuilder(self.component_system, "multi_zone")
        builder.column = self.column
        builder.add_unit(source_serial)
        builder.add_unit(source_parallel)

        builder.add_unit(sink_serial)
        builder.add_unit(sink_parallel)

        builder.add_unit(serial_zone)
        builder.add_unit(parallel_zone)

        builder.add_connection(source_serial, serial_zone)
        builder.add_connection(serial_zone, sink_serial)
        builder.add_connection(serial_zone, parallel_zone)
        builder.set_output_state(serial_zone, [0.5, 0.5])

        builder.add_connection(source_parallel, parallel_zone)
        builder.add_connection(parallel_zone, sink_parallel)

        builder.switch_time = 300

        return builder

    def test_units(self):
        """Check if all units are added properly in the FlowSheet"""
        # Serial
        builder = self.create_serial()
        flow_sheet = builder.build_flow_sheet()

        units_expected = [
            'source', 'sink',
            'serial_inlet', 'serial_outlet',
            'column_upstream_0', 'column_downstream_0',
            'column_upstream_1', 'column_downstream_1'
        ]

        self.assertEqual(units_expected, flow_sheet.unit_names)

        # Parallel
        builder = self.create_parallel()
        flow_sheet = builder.build_flow_sheet()

        units_expected = [
            'source', 'sink',
            'parallel_inlet', 'parallel_outlet',
            'column_upstream_0', 'column_downstream_0',
            'column_upstream_1', 'column_downstream_1'
        ]

        self.assertEqual(units_expected, flow_sheet.unit_names)

        # SMB
        builder = self.create_smb()
        flow_sheet = builder.build_flow_sheet()

        units_expected = [
            'feed', 'eluent',
            'raffinate', 'extract',
            'zone_I_inlet', 'zone_I_outlet',
            'column_upstream_0', 'column_downstream_0',
            'zone_II_inlet', 'zone_II_outlet',
            'column_upstream_1', 'column_downstream_1',
            'zone_III_inlet', 'zone_III_outlet',
            'column_upstream_2', 'column_downstream_2',
            'zone_IV_inlet', 'zone_IV_outlet',
            'column_upstream_3', 'column_downstream_3'
        ]

        self.assertEqual(units_expected, flow_sheet.unit_names)

    def test_connections(self):
        """Check if all units are connected properly in the FlowSheet"""
        # Serial
        builder = self.create_serial()
        flow_sheet = builder.build_flow_sheet()
        serial_zone = builder.zones[0]

        self.assertTrue(
            flow_sheet.connection_exists('source', serial_zone.inlet_unit.name),
            msg='Serial zone inlet missing'
            )

        # Serial zone should connect to all column tops
        for col in builder.columns:
            with self.subTest(zone='serial', connection='zone inlet to column', column=col.index):
                self.assertTrue(
                    flow_sheet.connection_exists(
                        serial_zone.inlet_unit.name,
                        col.top.name
                    ),
                    msg=f'{serial_zone.inlet_unit.name} connection to {col.top.name}'
                    )

        # Column subunits should be chained in order
        for col in builder.columns:
            with self.subTest(zone='serial', connection='subunit chain', column=col.index):
                for upstream, downstream in zip(col.subunits, col.subunits[1:]):
                    self.assertTrue(
                        flow_sheet.connection_exists(upstream.name, downstream.name),
                        msg=f'{upstream.name} connection to {downstream.name}'
                        )

        # Bottom of each column should connect to top of next
        cols = builder.columns
        for this_col, next_col in zip(cols, cols[1:] + cols[:1]):
            with self.subTest(zone='serial', connection='column chain', column=this_col.index):
                self.assertTrue(flow_sheet.connection_exists(
                    this_col.bottom, next_col.top
                    ),
                    msg=f'{this_col.bottom.name} connection to {next_col.top.name}'
                    )

        # Bottom of each column should connect to zone outlet
        for col in builder.columns:
            with self.subTest(zone='serial', connection='column to zone outlet', column=col.index):
                self.assertTrue(
                    flow_sheet.connection_exists(
                        col.bottom.name,
                        serial_zone.outlet_unit.name
                    ),
                    msg=f'{serial_zone.outlet_unit.name} connection to {col.bottom.name}'
                )

        # Parallel
        builder = self.create_parallel()
        flow_sheet = builder.build_flow_sheet()
        parallel_zone = builder.zones[0]

        self.assertTrue(
            flow_sheet.connection_exists('source', parallel_zone.inlet_unit.name),
            msg='Parallel zone inlet missing'
        )

        # Parallel zone inlet should connect to all column tops
        for col in builder.columns:
            with self.subTest(zone='parallel', connection='zone inlet to column', column=col.index):
                self.assertTrue(
                    flow_sheet.connection_exists(
                        parallel_zone.inlet_unit.name,
                        col.top.name
                    ),
                    msg=f'{parallel_zone.inlet_unit.name} connection to {col.top.name}'
                )

        # Column subunits should be chained in order
        for col in builder.columns:
            with self.subTest(zone='parallel', connection='subunit chain', column=col.index):
                for upstream, downstream in zip(col.subunits, col.subunits[1:]):
                    self.assertTrue(
                        flow_sheet.connection_exists(upstream.name, downstream.name),
                        msg=f'{upstream.name} connection to {downstream.name}'
                        )

        # Bottom of each column should connect to zone outlet
        for col in builder.columns:
            with self.subTest(
                zone='parallel',
                connection='column to zone outlet',
                column=col.index
                ):
                self.assertTrue(
                    flow_sheet.connection_exists(
                        col.bottom.name,
                        parallel_zone.outlet_unit.name
                    ),
                    msg=f'{parallel_zone.outlet_unit.name} connection to {col.bottom.name}'
                )

        # SMB
        builder = self.create_smb()
        flow_sheet = builder.build_flow_sheet()
        zones = builder.zones
        cols = builder.columns

        self.assertTrue(flow_sheet.connection_exists('eluent', 'zone_I_inlet'))
        self.assertTrue(flow_sheet.connection_exists('feed', 'zone_III_inlet'))

        self.assertTrue(flow_sheet.connection_exists("zone_I_outlet", "extract"))
        self.assertTrue(flow_sheet.connection_exists("zone_I_outlet", "zone_II_inlet"))

        self.assertTrue(flow_sheet.connection_exists("zone_I_outlet", "extract"))
        self.assertTrue(flow_sheet.connection_exists("zone_I_outlet", "zone_II_inlet"))

        self.assertTrue(
            flow_sheet.connection_exists("zone_II_outlet", "zone_III_inlet")
        )

        # Each zone inlet should connect to top of each column
        for zone in zones:
            for col in cols:
                with self.subTest(
                    zone=zone.name,
                    connection='zone inlet to column top',
                    column=col.index
                ):
                    self.assertTrue(
                        flow_sheet.connection_exists(
                            zone.inlet_unit.name, col.top.name
                            ),
                        msg=f'{zone.name} connection to {col.top.name}'
                        )

        # Bottom of each column should connect to each zone outlet
        for zone in zones:
            for col in cols:
                with self.subTest(
                    zone=zone.name,
                    connection='column bottom to zone outlet',
                    column=col.index
                ):
                    self.assertTrue(
                        flow_sheet.connection_exists(col.bottom.name, zone.outlet_unit.name),
                        msg=f'{col.bottom.name} connection to {zone.name}'
                    )

    def test_column_position_indices(self):
        """Test column position indices."""
        builder = self.create_smb()

        # Initial state, position 0
        carousel_position = 0
        carousel_state = 0
        indices_expected = 0

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

        # Initial state, position 1
        carousel_position = 1
        carousel_state = 0
        indices_expected = 1

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

        # First state, position 0
        carousel_position = 0
        carousel_state = 1
        indices_expected = 1

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

        # First state, position 1
        carousel_position = 1
        carousel_state = 1
        indices_expected = 2

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

        # 4th state (back to initial state), position 0
        carousel_position = 0
        carousel_state = 4
        indices_expected = 0

        indices = builder.column_indices_at_state(carousel_position, carousel_state)
        self.assertEqual(indices_expected, indices)

        time = carousel_state * builder.switch_time
        indices = builder.column_indices_at_time(time, carousel_position)
        self.assertEqual(indices_expected, indices)

    def test_carousel_state(self):
        """Test carousel state."""

        builder = self.create_smb()

        # Initial state
        time = 0
        state_expected = 0

        state = builder.carousel_state(time)

        self.assertEqual(state_expected, state)

        # Position 0
        time = builder.switch_time / 2
        state_expected = 0

        state = builder.carousel_state(time)

        self.assertEqual(state_expected, state)

        # Position 1
        time = builder.switch_time
        state_expected = 1

        state = builder.carousel_state(time)

        self.assertEqual(state_expected, state)

        # Back to initial state; position 0
        time = 4 * builder.switch_time
        state_expected = 0

        state = builder.carousel_state(time)

        self.assertEqual(state_expected, state)

    def test_flow_rates(self):
        """Test flow rate splits and sums."""
        # Serial
        builder = self.create_serial()
        process = builder.build_process()
        serial_zone = builder.zones[0]
        t0 = 0.0

        # Inlet and outlet are almost equal
        for port in (serial_zone.inlet_unit.name, serial_zone.outlet_unit.name):
            with self.subTest(zone='serial', port=port):
                flow_rate = process.flow_rate_timelines[port].total_in[None].value(t0)
                np.testing.assert_almost_equal(flow_rate, 2e-7)

        # Each column bottom sees full flow
        for col in builder.columns:
            with self.subTest(zone='serial', column=col.index):
                name = col.bottom.name
                flow_rate = process.flow_rate_timelines[name].total_in[None].value(t0)
                np.testing.assert_almost_equal(flow_rate, 2e-7)

        # Parallel (flow is split between columns)
        builder = self.create_parallel()
        process = builder.build_process()
        parallel_zone = builder.zones[0]
        t0 = 0.0

        # Inlet and outlet are almost equal
        for port in (parallel_zone.inlet_unit.name, parallel_zone.outlet_unit.name):
            with self.subTest(zone='parallel', port=port):
                flow_rate = process.flow_rate_timelines[port].total_in[None].value(t0)
                np.testing.assert_almost_equal(flow_rate, 2e-7)

        # Each column bottom sees equal split
        share = 2e-7 / len(builder.columns)
        for col in builder.columns:
            with self.subTest(zone='parallel', column=col.index):
                name = col.bottom.name
                flow_rate = process.flow_rate_timelines[name].total_in[None].value(t0)
                np.testing.assert_almost_equal(flow_rate, share)

        # Multi-Zone (Side streams between zones)
        builder = self.create_multi_zone()
        process = builder.build_process()

        serial_inlet = process.flow_rate_timelines['serial_inlet']
        parallel_inlet = process.flow_rate_timelines['parallel_inlet']
        column_0 = process.flow_rate_timelines['column_downstream_0']
        column_2 = process.flow_rate_timelines['column_downstream_2']

        flow_rate = serial_inlet.total_in[None].value(0)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = parallel_inlet.total_in[None].value(0)
        flow_rate_expected = 3e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        # Initial state
        t = 0
        flow_rate = column_0.total_in[None].value(t)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(
            flow_rate, flow_rate_expected,)

        flow_rate = column_2.total_in[None].value(t)
        flow_rate_expected = 1.5e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        # First position
        t = builder.switch_time
        flow_rate = column_0.total_in[None].value(t)
        flow_rate_expected = 1.5e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

        flow_rate = column_2.total_in[None].value(t)
        flow_rate_expected = 2e-7
        np.testing.assert_almost_equal(flow_rate, flow_rate_expected)

    def test_flow_direction(self):
        """Test flow directions in a multizone configuration."""
        # Multi-Zone (Side streams between zones)
        builder = self.create_multi_zone()
        process = builder.build_process()

        checks = [
            # (column index, time, expected flow_direction)
            (0, 0, 1),
            (2, 0, -1),
            (0, builder.switch_time, -1),
            (2, builder.switch_time, 1)
        ]

        for col_index, t, expected in checks:
            bottom_name = builder.columns[col_index].bottom.name
            path = f'flow_sheet.{bottom_name}.flow_direction'

            with self.subTest(column=col_index, time=t, unit=bottom_name):
                tl = process.parameter_timelines[path]
                flow_direction = tl.value(t)
                np.testing.assert_almost_equal(
                    flow_direction, expected
                    )

    def test_single_subunit_column(self):
        """Test a system with a single column subunit."""
        builder = self.create_serial()
        builder.column = LumpedRateModelWithoutPores(self.component_system, name='column')
        flow_sheet = builder.build_flow_sheet()

        # Check that n columns exist
        self.assertEqual(len(builder.columns), builder.n_columns)

        serial_zone = builder.zones[0]
        cols = builder.columns

        # Each column has a single subunit
        for col in cols:
            with self.subTest(column=col.index):
                self.assertEqual(len(col.subunits), 1)

        # No intracolumn chaining exists
        for col in cols:
            name = col.subunits[0].name
            with self.subTest(column=col.index):
                self.assertFalse(
                    flow_sheet.connection_exists(name, name),
                    msg=f'Unexpected self-chain on {name}'
                )

        # Zone inlet connects to subunit, subunit to zone outlet
        for col in cols:
            sub = col.subunits[0]
            with self.subTest(column=col.index, connection='zone inlet to column'):
                self.assertTrue(
                    flow_sheet.connection_exists(
                        serial_zone.inlet_unit.name,
                        sub.name
                    ),
                    msg=f'{serial_zone.inlet_unit.name} connection to {sub.name}'
                )
                self.assertTrue(
                    flow_sheet.connection_exists(
                        sub.name,
                        serial_zone.outlet_unit.name
                    ),
                    msg=f'{sub.name} connection to {serial_zone.outlet_unit.name}'
                )

        # Columns are chained in order
        for this_col, next_col in zip(cols, cols[1:] + cols[:1]):
            with self.subTest(connection='column chain', column=this_col.index):
                this_col_name = this_col.subunits[0].name
                next_col_name = next_col.subunits[0].name
                self.assertTrue(
                    flow_sheet.connection_exists(
                        this_col_name, next_col_name
                    ),
                    msg=f'{this_col_name} connection to {next_col_name}'
                )

    def test_simulation(self):
        builder = self.create_serial()
        process = builder.build_process()

        process_simulator = Cadet()
        simulation_results = process_simulator.simulate(process)

        self.assertEqual(simulation_results.exit_flag, 0)


class Test_SerialCarousel(unittest.TestCase):
    """
    Test suite to handle the specific implementations for the SerialCarouselBuilder.
    """
    def setUp(self):
        self.component_system = ComponentSystem(2)

        self.binding_model = Linear(self.component_system)
        self.binding_model.adsorption_rate = [6, 8]
        self.binding_model.desorption_rate = [1, 1]

        self.column = [
            LumpedRateModelWithoutPores(self.component_system, name='upstream'),
            LumpedRateModelWithoutPores(self.component_system, name='downstream')
            ]
        for subunit in self.column:
            subunit.length = 0.6
            subunit.diameter = 0.024
            subunit.axial_dispersion = 4.7e-7
            subunit.total_porosity = 0.7

            subunit.binding_model = self.binding_model

        self.pipe = LumpedRateModelWithoutPores(self.component_system, name='pipe')
        
    def create_carousel(self, with_pipe:bool):
        source = Inlet(self.component_system, name='feed')
        sink = Outlet(self.component_system, name='raffinate')
        serial_zone = SerialZone(self.component_system, 'serial_zone', 3)

        builder = SerialCarouselBuilder(self.component_system, 'serial_carousel')
        builder.column = self.column
        if with_pipe:
            builder.pipe = self.pipe
        
        builder.add_unit(source)
        builder.add_unit(sink)
        builder.add_unit(serial_zone)

        builder.add_connection(source, serial_zone)
        builder.add_connection(serial_zone, sink)

        builder.switch_time = 300

        return builder, serial_zone
    
    def test_units_and_pipes(self):
        """Check if units are created, and pipes if passed."""
        for mode in (False, True):
            builder, _ = self.create_carousel(with_pipe=mode)
            flow_sheet = builder.build_flow_sheet()

            # Expect 3 columns * 2 subunits = 6 units total
            n_cols = builder.n_columns
            n_col_units_expected =  n_cols * len(builder.column)
            n_pipes_expected = n_cols if mode else 0  # system with n cols has n pipes
            self.assertEqual(
                len([u for u in flow_sheet.units if 'pipe_' in u.name]),
                n_pipes_expected
            )
            self.assertEqual(
                len([u for u in flow_sheet.units if 'column_' in u.name]),
                n_col_units_expected
            )
    
    def test_ring_connections(self):
        """Check if column bottom -> pipe -> column top."""
        builder, _ = self.create_carousel(with_pipe=True)
        flow_sheet = builder.build_flow_sheet()
        for this_col, next_col in zip(
            builder.columns, builder.columns[1:] + builder.columns[:1]
        ):
            pipe_name = f'pipe_{this_col.index}_{next_col.index}'
            self.assertTrue(
                flow_sheet.connection_exists(
                    this_col.bottom.name, pipe_name
                )
            )
            self.assertTrue(
                pipe_name, next_col.top.name
            )

        builder, _ = self.create_carousel(with_pipe=False)
        flow_sheet = builder.build_flow_sheet()
        for this_col, next_col in zip(
            builder.columns, builder.columns[1:] + builder.columns[:1]
        ):
            self.assertTrue(
                flow_sheet.connection_exists(
                    this_col.bottom.name, next_col.top.name
                )
            )
    
    def test_pipe_initial_state(self):
        """Check that pipe initial states are set to the owning zone."""
        builder, zone = self.create_carousel(with_pipe=True)
        zone.initial_state = [
            {'c': [1, 1]}, {'c': [2, 2]}, {'c': [3, 3]}
        ]
        flow_sheet = builder.build_flow_sheet()
        for this_col, next_col in zip(
            builder.columns, builder.columns[1:] + builder.columns[:1]
        ):
            pipe = flow_sheet[f'pipe_{this_col.index}_{next_col.index}']
            expected = zone.initial_state[this_col.index]['c']
            self.assertAlmostEqual(
                pipe.initial_state['c'], expected
            )
    
    def test_event_targets(self):
        """Check that events route the correct ports."""
        builder, _ = self.create_carousel(with_pipe=True)
        process = builder.build_process()

        event_names = {e.name:e for e in process.events}

        # First column
        evt = event_names['column_0_0']
        self.assertEqual(
            evt.parameter_path,
            'flow_sheet.output_states.column_downstream_0'
        )
        self.assertEqual(evt.state, builder.n_zones) # should be routed to last port
        
        # Last column
        evt = event_names['column_2_0']
        self.assertEqual(
            evt.parameter_path,
            'flow_sheet.output_states.column_downstream_2'
            )
        self.assertEqual(evt.state, 0)  # should be routed to zone outlet (first port)

    def test_simulate(self):
        builder, _ = self.create_carousel(with_pipe=False)
        process = builder.build_process()
        process_simulator = Cadet()
        simulation_results = process_simulator.simulate(process)

        self.assertEqual(simulation_results.exit_flag, 0)


class Test_SerialCarousel(unittest.TestCase):
    """
    Test suite to handle the specific implementations for the SerialCarouselBuilder.
    """
    def setUp(self):
        self.component_system = ComponentSystem(2)

        self.binding_model = Linear(self.component_system)
        self.binding_model.adsorption_rate = [6, 8]
        self.binding_model.desorption_rate = [1, 1]

        self.column = [
            LumpedRateModelWithoutPores(self.component_system, name='upstream'),
            LumpedRateModelWithoutPores(self.component_system, name='downstream')
            ]
        for subunit in self.column:
            subunit.length = 0.6
            subunit.diameter = 0.024
            subunit.axial_dispersion = 4.7e-7
            subunit.total_porosity = 0.7

            subunit.binding_model = self.binding_model

        self.pipe = LumpedRateModelWithoutPores(self.component_system, name='pipe')
        
    def create_carousel(self, with_pipe:bool):
        source = Inlet(self.component_system, name='feed')
        sink = Outlet(self.component_system, name='raffinate')
        serial_zone = SerialZone(self.component_system, 'serial_zone', 3)

        builder = SerialCarouselBuilder(self.component_system, 'serial_carousel')
        builder.column = self.column
        if with_pipe:
            builder.pipe = self.pipe
        
        builder.add_unit(source)
        builder.add_unit(sink)
        builder.add_unit(serial_zone)

        builder.add_connection(source, serial_zone)
        builder.add_connection(serial_zone, sink)

        builder.switch_time = 300

        return builder, serial_zone
    
    def test_units_and_pipes(self):
        """Check if units are created, and pipes if passed."""
        for mode in (False, True):
            builder, _ = self.create_carousel(with_pipe=mode)
            flow_sheet = builder.build_flow_sheet()

            # Expect 3 columns * 2 subunits = 6 units total
            n_cols = builder.n_columns
            n_col_units_expected =  n_cols * len(builder.column)
            n_pipes_expected = n_cols if mode else 0  # system with n cols has n pipes
            self.assertEqual(
                len([u for u in flow_sheet.units if 'pipe_' in u.name]),
                n_pipes_expected
            )
            self.assertEqual(
                len([u for u in flow_sheet.units if 'column_' in u.name]),
                n_col_units_expected
            )
    
    def test_ring_connections(self):
        """Check if column bottom -> pipe -> column top."""
        builder, _ = self.create_carousel(with_pipe=True)
        flow_sheet = builder.build_flow_sheet()
        for this_col, next_col in zip(
            builder.columns, builder.columns[1:] + builder.columns[:1]
        ):
            pipe_name = f'pipe_{this_col.index}_{next_col.index}'
            self.assertTrue(
                flow_sheet.connection_exists(
                    this_col.bottom.name, pipe_name
                )
            )
            self.assertTrue(
                flow_sheet.connection_exists(
                pipe_name, next_col.top.name
                )
            )

        builder, _ = self.create_carousel(with_pipe=False)
        flow_sheet = builder.build_flow_sheet()
        for this_col, next_col in zip(
            builder.columns, builder.columns[1:] + builder.columns[:1]
        ):
            self.assertTrue(
                flow_sheet.connection_exists(
                    this_col.bottom.name, next_col.top.name
                )
            )
    
    def test_pipe_initial_state(self):
        """Check that pipe initial states are set to the owning zone."""
        builder, zone = self.create_carousel(with_pipe=True)
        zone.initial_state = [
            {'c': [1, 1]}, {'c': [2, 2]}, {'c': [3, 3]}
        ]
        flow_sheet = builder.build_flow_sheet()
        for this_col, next_col in zip(
            builder.columns, builder.columns[1:] + builder.columns[:1]
        ):
            pipe = flow_sheet[f'pipe_{this_col.index}_{next_col.index}']
            expected = zone.initial_state[this_col.index]['c']
            self.assertEqual(
                pipe.initial_state['c'], expected
            )
    
    def test_event_targets(self):
        """Check that events route the correct ports."""
        builder, _ = self.create_carousel(with_pipe=True)
        process = builder.build_process()

        event_names = {e.name:e for e in process.events}

        # First column
        evt = event_names['column_0_0']
        self.assertEqual(
            evt.parameter_path,
            'flow_sheet.output_states.column_downstream_0'
        )
        self.assertEqual(evt.state, builder.n_zones) # should be routed to last port
        
        # Last column
        evt = event_names['column_2_0']
        self.assertEqual(
            evt.parameter_path,
            'flow_sheet.output_states.column_downstream_2'
            )
        self.assertEqual(evt.state, 0)  # should be routed to zone outlet (first port)

    def test_simulate(self):
        builder, _ = self.create_carousel(with_pipe=False)
        process = builder.build_process()
        process_simulator = Cadet()
        simulation_results = process_simulator.simulate(process)

        self.assertEqual(simulation_results.exit_flag, 0)


if __name__ == "__main__":
    unittest.main()
