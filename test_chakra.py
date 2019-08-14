import pytest
import copy
import numpy as np
from numpy.testing import assert_allclose

# Local imports
import oggm
from oggm.core.massbalance import LinearMassBalance
from oggm import utils, cfg
from oggm.cfg import SEC_IN_DAY
from oggm.core.sia2d import Upstream2D
from oggm.exceptions import InvalidParamsError

# Tests
from oggm.tests.funcs import (dummy_bumpy_bed, dummy_constant_bed,
                              dummy_constant_bed_cliff,
                              dummy_mixed_bed, dummy_constant_bed_obstacle,
                              dummy_noisy_bed, dummy_parabolic_bed,
                              dummy_trapezoidal_bed, dummy_width_bed,
                              dummy_width_bed_tributary,
                              patch_url_retrieve_github)

import matplotlib.pyplot as plt
from oggm.core.flowline import (KarthausModel, FluxBasedModel,
                                MUSCLSuperBeeModel, MassConservationChecker)

from chakra import (CalvingModel, WaterMassBalance, FixedMassBalance,
                    dummy_tidewater_bed, find_sia_flux_from_thickness)

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
    models = [MUSCLSuperBeeModel, CalvingModel]

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
    np.testing.assert_allclose(volume[0][-1], volume[2][-1], atol=3e-3)
    np.testing.assert_allclose(volume[1][-1], volume[2][-1], atol=3e-3)

    assert utils.rmsd(lens[0], lens[1]) < 50.
    assert utils.rmsd(volume[0], volume[1]) < 2e-3
    assert utils.rmsd(surface_h[0], surface_h[1]) < 1.0


def test_find_flux_from_thickness():

    mb = LinearMassBalance(2600.)
    model = CalvingModel(dummy_constant_bed(), mb_model=mb)
    model.run_until(700)

    # Pick a flux and slope somewhere in the glacier
    for i in [1, 10, 20, 50]:
        flux = model.flux_stag[0][i]
        slope = model.slope_stag[0][i]
        thick = model.thick_stag[0][i]
        width = model.fls[0].widths_m[i]

        out = find_sia_flux_from_thickness(slope, width, thick)
        assert_allclose(out, flux, atol=1e-7)


def test_fixed_massbalance():

    mb_mod = FixedMassBalance()
    to_test = mb_mod.get_annual_mb([200, 100, 0, -1, -100])
    # Convert units
    to_test *= cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
    # Test
    assert_allclose(to_test, [0, 0, 0, 0, 0])

    mb_mod = FixedMassBalance(1000.)
    to_test = mb_mod.get_annual_mb([200, 100])
    # Convert units
    to_test *= cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']
    # Test
    assert_allclose(to_test, [1000, 1000])


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


def test_bed():

    fl = dummy_tidewater_bed()[-1]

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

    fls = dummy_constant_bed()
    mb_mod = FixedMassBalance()

    model = CalvingModel(fls, mb_model=mb_mod, flux_gate_thickness=150,
                         check_for_boundaries=False)
    model.run_until(3000)

    df = model.get_diagnostics()
    assert_allclose(df['ice_thick'], 150, atol=1)
    assert_allclose(df['ice_flux'], df['ice_flux'].iloc[0], atol=1)

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

    fls = dummy_tidewater_bed()
    mb_mod = FixedMassBalance()

    model = CalvingModel(fls, mb_model=mb_mod, flux_gate=0.07,
                         fs=5.7e-20*4,  # quite slidy
                         )

    # Up to a certain stage its OK
    _, ds = model.run_until_and_store(6000)

    df = model.get_diagnostics()

    if do_plot:
        fl = model.fls[-1]
        x = fl.dis_on_line * fl.dx_meter

        plt.figure()
        (df[['ice_flux']] / cfg.SEC_IN_YEAR).plot()

        plt.figure()
        df[['u']].plot()

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

    # we check that a glacier going into water and melting will be different

    fls = dummy_tidewater_bed()

    # This is zero MB until water, then melt quite a lot
    mb_mod = WaterMassBalance(ela_h=0, grad=2, max_mb=0,
                              underwater_melt=-2000)

    model = CalvingModel(fls, mb_model=mb_mod, flux_gate=0.07,
                         fs=5.7e-20*4,  # quite slidy
                         )

    # Up to 2000 years is OK
    _, ds = model.run_until_and_store(8000)
    df = model.get_diagnostics()

    if do_plot:
        fl = model.fls[-1]
        x = fl.dis_on_line * fl.dx_meter

        plt.figure()
        (df[['ice_flux']] / cfg.SEC_IN_YEAR).plot()

        plt.figure()
        df[['u']].plot()

        plt.figure()
        ds.volume_m3.plot()

        plt.figure()
        plt.plot(x, fl.bed_h, 'k')
        plt.plot(x, fl.surface_h, 'C3')
        plt.hlines(0, 0, 6e4, color='C0')
        plt.ylim(-800, 1200)
        plt.xlabel('[m]')
        plt.ylabel('Elevation [m]')
        plt.savefig('/home/mowglie/glacier_melt.png')
        plt.show()
