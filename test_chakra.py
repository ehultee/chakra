import pytest
import numpy as np
from numpy.testing import assert_allclose

import matplotlib.pyplot as plt


# Local imports
from oggm.core.massbalance import LinearMassBalance, ScalarMassBalance
from oggm import utils, cfg

# Tests
from oggm.tests.funcs import dummy_constant_bed
from oggm.tests.ext.sia_fluxlim import MUSCLSuperBeeModel

from chakra import (KCalvingModel, WaterMassBalance, bu_tidewater_bed, ChakraModel)

# set to true to see what's going on
do_plot = True


def setup_module(module):
    """Used by pytest"""
    cfg.initialize()


def teardown_module(module):
    """Used by pytest"""
    pass


def test_numerics():

    # We test that our model produces similar results than Jarosh et al 2013.
    models = [MUSCLSuperBeeModel, ChakraModel]

    lens = []
    surface_h = []
    volume = []
    yrs = np.arange(1, 700, 2)
    for model_class in models:

        mb = LinearMassBalance(2600.)
        model = model_class(dummy_constant_bed(), mb_model=mb)

        length = yrs * 0.
        vol = yrs * 0.
        for i, y in enumerate(yrs):
            model.run_until(y)
            assert model.yr == y
            length[i] = model.fls[-1].length_m
            vol[i] = model.fls[-1].volume_km3
        lens.append(length)
        volume.append(vol)
        surface_h.append(model.fls[-1].surface_h.copy())

        # We are almost at equilibrium. Spec MB should be close to 0
        assert_allclose(mb.get_specific_mb(fls=model.fls), 0, atol=10)

    if do_plot:
        plt.figure()
        plt.plot(yrs, lens[0])
        plt.plot(yrs, lens[1])
        plt.title('Compare Length')
        plt.xlabel('years')
        plt.ylabel('[m]')
        plt.legend(['MUSCL-SuperBee', 'Chakra'], loc=2)

        plt.figure()
        plt.plot(yrs, volume[0])
        plt.plot(yrs, volume[1])
        plt.title('Compare Volume')
        plt.xlabel('years')
        plt.ylabel('[km^3]')
        plt.legend(['MUSCL-SuperBee', 'Chakra'], loc=2)

        plt.figure()
        plt.plot(model.fls[-1].bed_h, 'k')
        plt.plot(surface_h[0])
        plt.plot(surface_h[1])
        plt.title('Compare Shape')
        plt.xlabel('[m]')
        plt.ylabel('Elevation [m]')
        plt.legend(['Bed', 'MUSCL-SuperBee', 'Chakra'], loc=2)
        plt.show()

    np.testing.assert_almost_equal(lens[0][-1], lens[1][-1])
    np.testing.assert_allclose(volume[0][-1], volume[1][-1], atol=1e-3)

    assert utils.rmsd(lens[0], lens[1]) < 50.
    assert utils.rmsd(volume[0], volume[1]) < 2e-3
    assert utils.rmsd(surface_h[0], surface_h[1]) < 1.0


def test_water_massbalance():

    mb_mod = LinearMassBalance(ela_h=100, grad=1)
    to_test = mb_mod.get_annual_mb([200, 100, 0, -1, -100])
    # Convert units
    to_test *= cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
    # Test
    assert_allclose(to_test, [100, 0, -100, -101, -200])

    mb_mod = WaterMassBalance(ela_h=100, grad=1)
    to_test = mb_mod.get_annual_mb([200, 100, 0, -1, -100])
    # Convert units
    to_test *= cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
    # Test
    assert_allclose(to_test, [100, 0, -100, 0, 0])

    mb_mod = WaterMassBalance(ela_h=100, grad=1, underwater_melt=-1000)
    to_test = mb_mod.get_annual_mb([200, 100, 0, -1, -100])
    # Convert units
    to_test *= cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
    # Test
    assert_allclose(to_test, [100, 0, -100, -1000, -1000])


def test_bu_bed():

    fl = bu_tidewater_bed()[-1]

    x = fl.dis_on_line * fl.dx_meter

    assert x[-1] == 6e4
    assert x[0] == 0
    assert len(fl.surface_h) == 201
    assert fl.dx_meter == 300

    if do_plot:
        plt.figure()
        plt.hlines(0, 0, 6e4, color='C0')
        plt.plot(x, fl.bed_h, color='k')
        plt.ylim(-350, 800)
        plt.show()


def test_flux_gate():

    # This is to check that we are conserving mass on a slab
    # There are more tests in OGGM proper, this is just to play around
    fls = dummy_constant_bed()
    mb_mod = ScalarMassBalance()

    model = ChakraModel(fls, mb_model=mb_mod, flux_gate_thickness=150,
                        check_for_boundaries=False)
    model.run_until(3000)

    df = model.get_diagnostics()
    df['ice_flux'] *= cfg.SEC_IN_YEAR
    assert_allclose(df['ice_thick'], 150, atol=1)
    assert_allclose(df['ice_flux'], df['ice_flux'].iloc[0], atol=0.2)

    if do_plot:
        fl = model.fls[-1]
        x = fl.dis_on_line * fl.dx_meter

        plt.figure()
        plt.plot(x, fl.bed_h, 'k')
        plt.plot(x, fl.surface_h, 'C3')
        plt.xlabel('[m]')
        plt.ylabel('Elevation [m]')
        plt.show()


def test_no_calving_will_error():

    # we check that a glacier going into water without melting further
    # will eventually reach the domain boundary and error

    fls = bu_tidewater_bed()
    mb_mod = ScalarMassBalance()

    model = ChakraModel(fls, mb_model=mb_mod,
                        flux_gate=0.07,
                        fs=5.7e-20 * 4,  # quite slidy
                        do_calving=False,
                        )

    # Up to a certain stage its OK
    _, ds = model.run_until_and_store(6000)

    # Mass-conservation check
    assert_allclose(model.flux_gate_m3_since_y0, ds.volume_m3[-1])

    if do_plot:
        fl = model.fls[-1]
        x = fl.dis_on_line * fl.dx_meter
        df = model.get_diagnostics()

        plt.figure()
        df[['ice_flux']].plot()

        plt.figure()
        (df[['ice_velocity']] * cfg.SEC_IN_YEAR).plot()

        plt.figure()
        ds.volume_m3.plot()

        plt.figure()
        plt.plot(x, fl.bed_h, 'k')
        plt.plot(x, fl.surface_h, 'C3')
        plt.hlines(0, 0, 6e4, color='C0')
        plt.ylim(-800, 1200)
        plt.xlabel('[m]')
        plt.ylabel('Elevation [m]')
        plt.show()

    # But eventually it will reach boundary (set
    with pytest.raises(RuntimeError):
        _, ds = model.run_until_and_store(8000)


def test_underwater_melt():

    # we check that a glacier going into water and melting will look different

    fls = bu_tidewater_bed()

    # This is zero MB until water, then melt quite a lot
    mb_mod = WaterMassBalance(ela_h=0, grad=2, max_mb=0,
                              underwater_melt=-2000)

    model = ChakraModel(fls, mb_model=mb_mod,
                        flux_gate=0.07,
                        fs=5.7e-20 * 4,  # quite slidy
                        do_calving=False,
                        )

    # Up to 8000 years is OK
    _, ds = model.run_until_and_store(8000)

    # Mass-conservation check doesn't work here
    assert model.flux_gate_m3_since_y0 != ds.volume_m3[-1]

    # Glacier is much shorter
    model.length_m < 35000

    if do_plot:
        fl = model.fls[-1]
        x = fl.dis_on_line * fl.dx_meter
        df = model.get_diagnostics()

        plt.figure()
        df[['ice_flux']].plot()

        plt.figure()
        (df[['ice_velocity']] * cfg.SEC_IN_YEAR).plot()

        plt.figure()
        ds.volume_m3.plot()

        plt.figure()
        plt.plot(x, fl.bed_h, 'k')
        plt.plot(x, fl.surface_h, 'C3')
        plt.hlines(0, 0, 6e4, color='C0')
        plt.ylim(-800, 1200)
        plt.xlabel('[m]')
        plt.ylabel('Elevation [m]')
        plt.show()


def test_simple_chakra_param():

    # We make a very simple param for calving: when water depth is larger
    # than 100m, we simply bulk remove the ice as calving

    def simple_calving(model, dt):
        """This is the calving param we give to the model.

        It will be called at each time step.

        It needs to document:
        model.calving_m3_since_y0
        model.calving_rate_myr

        And very likely also wants to update model.section (otherwise
        nothing changes)
        """

        # We assume only one flowline (this is OK for now)
        fl = model.fls[-1]

        # Where to remove ice
        loc_remove = np.nonzero(fl.bed_h < -100)[0]
        # How much will we remove
        section = fl.section
        vol_removed = np.sum(section[loc_remove] * fl.dx_meter)
        # Effectively remove mass
        section[loc_remove] = 0

        # Updates so that our parameterization actually does something
        fl.section = section

        # Total calved volume
        model.calving_m3_since_y0 += vol_removed

        # The following is a very silly way to compute calving rate,
        # but the units are ok.

        # Calving rate in units of meter per time
        rate = vol_removed / fl.section[loc_remove[0] - 1]
        # To units of m per year
        model.calving_rate_myr = rate / dt * cfg.SEC_IN_YEAR

        # This is a way for the programmer to add an attribute - here dummy one
        try:
            model.number_of_times_called += 1
        except AttributeError:
            # this happens only the first time
            model.number_of_times_called = 1

    # See how it goes
    fls = bu_tidewater_bed()
    mb_mod = ScalarMassBalance()
    model = ChakraModel(fls, mb_model=mb_mod, flux_gate=0.07,
                        fs=5.7e-20 * 4,  # quite slidy
                        apply_parameterization=simple_calving,
                        )

    # Up to a certain stage its OK
    _, ds = model.run_until_and_store(4000)

    # Mass-conservation check
    assert_allclose(model.flux_gate_m3_since_y0,
                    ds.volume_m3[-1] + model.calving_m3_since_y0)

    # Calving rate check
    assert_allclose(ds.calving_rate_myr[-1], 26, atol=1)

    if do_plot:
        fl = model.fls[-1]
        x = fl.dis_on_line * fl.dx_meter
        df = model.get_diagnostics()

        plt.figure()
        df[['ice_flux']].plot()

        plt.figure()
        (df[['ice_velocity']] * cfg.SEC_IN_YEAR).plot()

        plt.figure()
        ds.volume_m3.plot()

        plt.figure()
        ds.calving_rate_myr.plot()

        plt.figure()
        plt.plot(x, fl.bed_h, 'k')
        plt.plot(x, fl.surface_h, 'C3')
        plt.hlines(0, 0, 6e4, color='C0')
        plt.ylim(-800, 1200)
        plt.xlabel('[m]')
        plt.ylabel('Elevation [m]')
        plt.show()
