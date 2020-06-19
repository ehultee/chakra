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


def test_chakra_method_1():

    # See how it goes
    fls = bu_tidewater_bed(split_flowline_before_water=5)

    mb_mod = ScalarMassBalance()

    flux_gate = 0.04
    cfl_number = 0.01
    glen_a = cfg.PARAMS['glen_a'] * 3

    cm = ChakraModel(fls, mb_model=mb_mod,
                     smooth_trib_influx=False,
                     cfl_number=cfl_number,
                     # enforce that all mass comes into first point
                     flux_gate=flux_gate,
                     # the default in OGGM is to add the fluxgate to each fl
                     calving_use_limiter=True,
                     glen_a=glen_a,
                     yield_strength=50e3,
                     )

    om = KCalvingModel(bu_tidewater_bed(), mb_model=mb_mod,
                       smooth_trib_influx=False,
                       cfl_number=cfl_number,
                       # enforce that all mass comes into first point
                       flux_gate=flux_gate,
                       # the default in OGGM is to add the fluxgate to each fl
                       calving_use_limiter=True,
                       glen_a=glen_a,
                       calving_k=0.2,
                       do_kcalving=True,
                       is_tidewater=True
                       )

    # Up to a certain stage its OK
    _, dsc = cm.run_until_and_store(4000)
    _, dso = om.run_until_and_store(4000)

    if do_plot:

        fl = bu_tidewater_bed()[0]
        xc = fl.dis_on_line * fl.dx_meter

        dfc = cm.get_diagnostics(fl_id=0)
        dfo = om.get_diagnostics(fl_id=0)

        plt.figure()
        f, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(xc, fl.bed_h, color='k')
        dfo['surface_h'].plot(ax=ax, color='C2', label='OGGM', legend=False)
        dfc['surface_h'].plot(ax=ax, color='C1', label='Chakra_OGGM', legend=False)
        plt.plot(cm.plastic_coord, cm.plastic_surface, 'C3', label='Chakra_SERMeQ')
        plt.hlines(0, *xc[[0, -1]], color='C0', linestyles=':')
        plt.ylim(-350, 1000);
        plt.ylabel('Altitude [m]');
        plt.xlabel('Distance along flowline [km]');
        plt.legend()
        plt.show()
