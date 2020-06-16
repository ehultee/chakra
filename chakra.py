"""Chakra: smooth, physically based calving in OGGM.

Authors: Lizz Ultee, Fabien Maussion & the OGGM developers
"""

# Builtins
import copy
from functools import partial

# External libs
import numpy as np
import pandas as pd

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm.exceptions import InvalidParamsError
from oggm.core.massbalance import LinearMassBalance, MassBalanceModel

# Constants
from oggm.cfg import SEC_IN_HOUR, SEC_IN_DAY, SEC_IN_YEAR, G
from oggm.core.flowline import (FlowlineModel, RectangularBedFlowline,
                                flux_gate_with_build_up)
from oggm.core.inversion import find_sia_flux_from_thickness


class WaterMassBalance(LinearMassBalance):
    """MB as a linear function of altitude, and a fixed value underwater.
    """

    def __init__(self, ela_h, grad=3., max_mb=None, underwater_melt=0):
        """ Initialize.

        Parameters
        ----------
        ela_h: float
            Equilibrium line altitude (units: [m])
        grad: float
            Mass-balance gradient (unit: [mm w.e. yr-1 m-1])
        max_mb: float
            Cap the mass balance to a certain value (unit: [mm w.e. yr-1])
        underwater_melt: float
            Fixed mass balance value (unit: [mm w.e. yr-1])
        """
        super(WaterMassBalance, self).__init__(ela_h, grad=grad, max_mb=max_mb)

        self.underwater_melt = underwater_melt / SEC_IN_YEAR / self.rho

    def get_monthly_mb(self, heights, year=None, fl_id=None, fls=None):
        heights = np.asarray(heights)
        mb = super(WaterMassBalance, self).get_monthly_mb(heights)
        return np.where(heights < 0, self.underwater_melt, mb)

    def get_annual_mb(self, heights, year=None, fl_id=None, fls=None):
        heights = np.asarray(heights)
        mb = super(WaterMassBalance, self).get_annual_mb(heights)
        return np.where(heights < 0, self.underwater_melt, mb)


def bu_tidewater_bed(gridsize=200, gridlength=6e4, widths_m=600,
                     b_0=260, alpha=0.017, b_1=350, x_0=4e4, sigma=1e4,
                     water_level=0, split_flowline_before_water=None):

    # Bassis & Ultee bed profile
    dx_meter = gridlength / gridsize
    x = np.arange(gridsize+1) * dx_meter
    bed_h = b_0 - alpha * x + b_1 * np.exp(-((x - x_0) / sigma)**2)
    bed_h += water_level
    surface_h = bed_h
    widths = surface_h * 0. + widths_m / dx_meter

    if split_flowline_before_water is not None:
        bs = np.min(np.nonzero(bed_h < 0)[0]) - split_flowline_before_water
        fls = [RectangularBedFlowline(dx=1, map_dx=dx_meter,
                                      surface_h=surface_h[:bs],
                                      bed_h=bed_h[:bs],
                                      widths=widths[:bs]),
               RectangularBedFlowline(dx=1, map_dx=dx_meter,
                                      surface_h=surface_h[bs:],
                                      bed_h=bed_h[bs:],
                                      widths=widths[bs:]),
               ]
        fls[0].set_flows_to(fls[1], check_tail=False, to_head=True)
        return fls
    else:
        return [
            RectangularBedFlowline(dx=1, map_dx=dx_meter, surface_h=surface_h,
                                   bed_h=bed_h, widths=widths)]


class KCalvingModel(FlowlineModel):
    """A fork/copy of the flowline model used by OGGM in production.

    We removed the shape factors to simplify the code a little, otherwise it's
    very close to OGGM.

    It solves for the SIA along the flowline(s) using a staggered grid. It
    computes the *ice flux* between grid points and transports the mass
    accordingly (also between flowlines).

    This model is numerically less stable than fancier schemes, but it
    is fast and works with multiple flowlines of any bed shape (rectangular,
    parabolic, trapeze, and any combination of them).

    We test that it conserves mass in most cases, but not on very stiff cliffs.
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., inplace=False, fixed_dt=None, cfl_number=None,
                 min_dt=None, flux_gate_thickness=None,
                 flux_gate=None, flux_gate_build_up=100,
                 do_kcalving=None, calving_k=None, calving_use_limiter=None,
                 calving_limiter_frac=None, water_level=None,
                 **kwargs):
        """Instanciate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBakanceModel
            the mass-balance model
        y0 : int
            initial year of the simulation
        glen_a : float
            Glen's creep parameter
        fs : float
            Oerlemans sliding parameter
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        fixed_dt : float
            set to a value (in seconds) to prevent adaptive time-stepping.
        cfl_number : float
            Defaults to cfg.PARAMS['cfl_number'].
            For adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx / max_u).
            To choose the "best" CFL number we would need a stability
            analysis - we used an empirical analysis (see blog post) and
            settled on 0.02 for the default cfg.PARAMS['cfl_number'].
        min_dt : float
            Defaults to cfg.PARAMS['cfl_min_dt'].
            At high velocities, time steps can become very small and your
            model might run very slowly. In production, it might be useful to
            set a limit below which the model will just error.
        is_tidewater: bool, default: False
            is this a tidewater glacier?
        is_lake_terminating: bool, default: False
            is this a lake terminating glacier?
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass-balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        flux_gate_thickness : float or array
            flux of ice from the left domain boundary (and tributaries).
            Units of m of ice thickness. Note that unrealistic values won't be
            met by the model, so this is really just a rough guidance.
            It's better to use `flux_gate` instead.
        flux_gate : float or function or array of floats or array of functions
            flux of ice from the left domain boundary (and tributaries)
            (unit: m3 of ice per second). If set to a high value, consider
            changing the flux_gate_buildup time. You can also provide
            a function (or an array of functions) returning the flux
            (unit: m3 of ice per second) as a function of time.
            This is overriden by `flux_gate_thickness` if provided.
        flux_gate_buildup : int
            number of years used to build up the flux gate to full value
        do_kcalving : bool
            switch on the k-calving parameterisation. Ignored if not a
            tidewater glacier. Use the option from PARAMS per default
        calving_k : float
            the calving proportionality constant (units: yr-1). Use the
            one from PARAMS per default
        calving_use_limiter : bool
            whether to switch on the calving limiter on the parameterisation
            makes the calving fronts thicker but the model is more stable
        calving_limiter_frac : float
            limit the front slope to a fraction of the calving front.
            "3" means 1/3. Setting it to 0 limits the slope to sea-level.
        water_level : float
            the water level. It should be zero m a.s.l, but:
            - sometimes the frontal elevation is unrealistically high (or low).
            - lake terminating glaciers
            - other uncertainties
            The default is 0. For lake terminating glaciers,
            it is inferred from PARAMS['free_board_lake_terminating'].
            The best way to set the water level for real glaciers is to use
            the same as used for the inversion (this is what
            `robust_model_run` does for you)
        """
        super(KCalvingModel, self).__init__(flowlines, mb_model=mb_model,
                                            y0=y0, glen_a=glen_a, fs=fs,
                                            inplace=inplace,
                                            water_level=water_level,
                                            **kwargs)

        self.fixed_dt = fixed_dt
        if min_dt is None:
            min_dt = cfg.PARAMS['cfl_min_dt']
        if cfl_number is None:
            cfl_number = cfg.PARAMS['cfl_number']
        self.min_dt = min_dt
        self.cfl_number = cfl_number

        # Calving params
        if do_kcalving is None:
            do_kcalving = cfg.PARAMS['use_kcalving_for_run']
        self.do_calving = do_kcalving and self.is_tidewater
        if calving_k is None:
            calving_k = cfg.PARAMS['calving_k']
        self.calving_k = calving_k / cfg.SEC_IN_YEAR
        if calving_use_limiter is None:
            calving_use_limiter = cfg.PARAMS['calving_use_limiter']
        self.calving_use_limiter = calving_use_limiter
        if calving_limiter_frac is None:
            calving_limiter_frac = cfg.PARAMS['calving_limiter_frac']
        if calving_limiter_frac > 0:
            raise NotImplementedError('calving limiter other than 0 not '
                                      'implemented yet')
        self.calving_limiter_frac = calving_limiter_frac

        # Flux gate
        self.flux_gate = utils.tolist(flux_gate, length=len(self.fls))
        self.flux_gate_m3_since_y0 = 0.
        if flux_gate_thickness is not None:
            # Compute the theoretical ice flux from the slope at the top
            flux_gate_thickness = utils.tolist(flux_gate_thickness,
                                               length=len(self.fls))
            self.flux_gate = []
            for fl, fgt in zip(self.fls, flux_gate_thickness):
                # We set the thickness to the desired value so that
                # the widths work ok
                fl = copy.deepcopy(fl)
                fl.thick = fl.thick * 0 + fgt
                slope = (fl.surface_h[0] - fl.surface_h[1]) / fl.dx_meter
                if slope == 0:
                    raise ValueError('I need a slope to compute the flux')
                flux = find_sia_flux_from_thickness(slope,
                                                    fl.widths_m[0],
                                                    fgt,
                                                    shape=fl.shape_str[0],
                                                    glen_a=self.glen_a,
                                                    fs=self.fs)
                self.flux_gate.append(flux)

        # convert the floats to function calls
        for i, fg in enumerate(self.flux_gate):
            if fg is None:
                continue
            try:
                # Do we have a function? If yes all good
                fg(self.yr)
            except TypeError:
                # If not, make one
                self.flux_gate[i] = partial(flux_gate_with_build_up,
                                            flux_value=fg,
                                            flux_gate_yr=(flux_gate_build_up +
                                                          self.y0))

        # Optim
        self.slope_stag = []
        self.thick_stag = []
        self.section_stag = []
        self.u_stag = []
        self.shapefac_stag = []
        self.flux_stag = []
        self.trib_flux = []
        for fl, trib in zip(self.fls, self._tributary_indices):
            nx = fl.nx
            # This is not staggered
            self.trib_flux.append(np.zeros(nx))
            # We add an additional fake grid point at the end of tributaries
            if trib[0] is not None:
                nx = fl.nx + 1
            # +1 is for the staggered grid
            self.slope_stag.append(np.zeros(nx + 1))
            self.thick_stag.append(np.zeros(nx + 1))
            self.section_stag.append(np.zeros(nx + 1))
            self.u_stag.append(np.zeros(nx + 1))
            self.shapefac_stag.append(np.ones(nx + 1))  # beware the ones!
            self.flux_stag.append(np.zeros(nx + 1))

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # Simple container
        mbs = []

        # Loop over tributaries to determine the flux rate
        for fl_id, fl in enumerate(self.fls):

            # This is possibly less efficient than zip() but much clearer
            trib = self._tributary_indices[fl_id]
            slope_stag = self.slope_stag[fl_id]
            thick_stag = self.thick_stag[fl_id]
            section_stag = self.section_stag[fl_id]
            sf_stag = self.shapefac_stag[fl_id]
            flux_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            u_stag = self.u_stag[fl_id]
            flux_gate = self.flux_gate[fl_id]

            # Flowline state
            surface_h = fl.surface_h
            thick = fl.thick
            section = fl.section
            dx = fl.dx_meter

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid point
            is_trib = trib[0] is not None
            if is_trib:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])
            elif self.do_calving and self.calving_use_limiter:
                # We lower the max possible ice deformation
                # by clipping the surface slope here. It is completely
                # arbitrary but reduces ice deformation at the calving front.
                # I think that in essence, it is also partly
                # a "calving process", because this ice deformation must
                # be less at the calving front. The result is that calving
                # front "free boards" are quite high.
                # Note that clipping to water_level is arbitrary,
                # it could be any value from surface to bed (bed h being
                # the default)
                surface_h = utils.clip_min(surface_h, self.water_level)

            # Staggered gradient
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Staggered thick
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            # _fd = 2/(N+2) * self.glen_a
            N = self.glen_n
            rhogh = (self.rho * G * slope_stag) ** N
            u_stag[:] = (thick_stag ** (
                        N + 1)) * self._fd * rhogh * sf_stag ** N + \
                        (thick_stag ** (N - 1)) * self.fs * rhogh

            # Staggered section
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flux_stag[:] = u_stag * section_stag

            # Add boundary condition
            if flux_gate is not None:
                flux_stag[0] = flux_gate(self.yr)

            # CFL condition
            if not self.fixed_dt:
                maxu = np.max(np.abs(u_stag))
                if maxu > cfg.FLOAT_EPS:
                    cfl_dt = self.cfl_number * dx / maxu
                else:
                    cfl_dt = dt

                # Update dt only if necessary
                if cfl_dt < dt:
                    dt = cfl_dt
                    if cfl_dt < self.min_dt:
                        raise RuntimeError(
                            'CFL error: required time step smaller '
                            'than the minimum allowed: '
                            '{:.1f}s vs {:.1f}s.'.format(cfl_dt, self.min_dt))

            # Since we are in this loop, reset the tributary flux
            trib_flux[:] = 0

            # We compute MB in this loop, before mass-redistribution occurs,
            # so that MB models which rely on glacier geometry to decide things
            # (like PyGEM) can do wo with a clean glacier state
            mbs.append(self.get_mb(fl.surface_h, self.yr,
                                   fl_id=fl_id, fls=self.fls))

        # Time step
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt

        # A second loop for the mass exchange
        for fl_id, fl in enumerate(self.fls):

            flx_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            tr = self._tributary_indices[fl_id]

            dx = fl.dx_meter

            is_trib = tr[0] is not None
            # For these we had an additional grid point
            if is_trib:
                flx_stag = flx_stag[:-1]

            # Mass-balance
            widths = fl.widths_m
            mb = mbs[fl_id]
            # Allow parabolic beds to grow
            mb = dt * mb * np.where((mb > 0.) & (widths == 0), 10., widths)

            # Update section with ice flow and mass balance
            new_section = (fl.section + (
                        flx_stag[0:-1] - flx_stag[1:]) * dt / dx +
                           trib_flux * dt / dx + mb)

            # Keep positive values only and store
            fl.section = utils.clip_min(new_section, 0)

            # Add the last flux to the tributary
            # this works because the lines are sorted in order
            if is_trib:
                # tr tuple: line_index, start, stop, gaussian_kernel
                self.trib_flux[tr[0]][tr[1]:tr[2]] += \
                    utils.clip_min(flx_stag[-1], 0) * tr[3]

            # If we use a flux-gate, store the total volume that came in
            self.flux_gate_m3_since_y0 += flx_stag[0] * dt

            # --- The rest is for calving only ---
            self.calving_rate_myr = 0.

            # If tributary, do calving only if we are not transferring mass
            if is_trib and flx_stag[-1] > 0:
                continue

            # No need to do calving in these cases either
            if not self.do_calving or not fl.has_ice():
                continue

            # We do calving only if the last glacier bed pixel is below water
            # (this is to avoid calving elsewhere than at the front)
            if fl.bed_h[fl.thick > 0][-1] > self.water_level:
                continue

            # We do calving only if there is some ice above wl
            last_above_wl = np.nonzero((fl.surface_h > self.water_level) &
                                       (fl.thick > 0))[0][-1]
            if fl.bed_h[last_above_wl] > self.water_level:
                continue

            # OK, we're really calving
            section = fl.section

            # Calving law
            h = fl.thick[last_above_wl]
            d = h - (fl.surface_h[last_above_wl] - self.water_level)
            k = self.calving_k
            q_calving = k * d * h * fl.widths_m[last_above_wl]
            # Add to the bucket and the diagnostics
            fl.calving_bucket_m3 += q_calving * dt
            self.calving_m3_since_y0 += q_calving * dt
            self.calving_rate_myr = (q_calving / section[last_above_wl] *
                                     cfg.SEC_IN_YEAR)

            # See if we have ice below sea-water to clean out first
            below_sl = (fl.surface_h < self.water_level) & (fl.thick > 0)
            to_remove = np.sum(section[below_sl]) * fl.dx_meter
            if 0 < to_remove < fl.calving_bucket_m3:
                # This is easy, we remove everything
                section[below_sl] = 0
                fl.calving_bucket_m3 -= to_remove
            elif to_remove > 0:
                # We can only remove part of if
                section[below_sl] = 0
                section[last_above_wl + 1] = (
                            (to_remove - fl.calving_bucket_m3)
                            / fl.dx_meter)
                fl.calving_bucket_m3 = 0

            # The rest of the bucket might calve an entire grid point
            vol_last = section[last_above_wl] * fl.dx_meter
            if fl.calving_bucket_m3 > vol_last:
                fl.calving_bucket_m3 -= vol_last
                section[last_above_wl] = 0

            # We update the glacier with our changes
            fl.section = section

        # Next step
        self.t += dt
        return dt

    def get_diagnostics(self, fl_id=-1):
        """Obtain model diagnostics in a pandas DataFrame.

        Parameters
        ----------
        fl_id : int
            the index of the flowline of interest, from 0 to n_flowline-1.
            Default is to take the last (main) one

        Returns
        -------
        a pandas DataFrame, which index is distance along flowline (m). Units:
            - surface_h, bed_h, ice_tick, section_width: m
            - section_area: m2
            - slope: -
            - ice_flux, tributary_flux: m3 of *ice* per second
            - ice_velocity: m per second (depth-section integrated)
        """

        fl = self.fls[fl_id]
        nx = fl.nx

        df = pd.DataFrame(index=fl.dx_meter * np.arange(nx))
        df.index.name = 'distance_along_flowline'
        df['surface_h'] = fl.surface_h
        df['bed_h'] = fl.bed_h
        df['ice_thick'] = fl.thick
        df['section_width'] = fl.widths_m
        df['section_area'] = fl.section

        # Staggered
        var = self.slope_stag[fl_id]
        df['slope'] = (var[1:nx + 1] + var[:nx]) / 2
        var = self.flux_stag[fl_id]
        df['ice_flux'] = (var[1:nx + 1] + var[:nx]) / 2
        var = self.u_stag[fl_id]
        df['ice_velocity'] = (var[1:nx + 1] + var[:nx]) / 2
        var = self.shapefac_stag[fl_id]
        df['shape_fac'] = (var[1:nx + 1] + var[:nx]) / 2

        # Not Staggered
        df['tributary_flux'] = self.trib_flux[fl_id]

        return df


class ChakraModel(KCalvingModel):
    """A sandbox model where Chakra will be implemented.

    It overrides the default model and simplifies it a bit to reduce the
    number of parameters, etc.
    """

    def __init__(self, flowlines, mb_model=None, y0=0.,
                 glen_a=None, fs=0.,
                 cfl_number=None,
                 flux_gate_thickness=None,
                 flux_gate=None,
                 flux_gate_build_up=100,
                 do_calving=True,
                 water_level=None,
                 is_tidewater=True,
                 apply_parameterization=None,
                 **kwargs):
        """Instanciate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBakanceModel
            the mass-balance model
        y0 : int
            initial year of the simulation
        glen_a : float
            Glen's creep parameter
        fs : float
            Oerlemans sliding parameter
        cfl_number : float
            Defaults to cfg.PARAMS['cfl_number'].
            For adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx / max_u).
            To choose the "best" CFL number we would need a stability
            analysis - we used an empirical analysis (see blog post) and
            settled on 0.02 for the default cfg.PARAMS['cfl_number'].
        is_tidewater: bool, default: False
            is this a tidewater glacier?
        is_lake_terminating: bool, default: False
            is this a lake terminating glacier?
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass-balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        flux_gate_thickness : float or array
            flux of ice from the left domain boundary (and tributaries).
            Units of m of ice thickness. Note that unrealistic values won't be
            met by the model, so this is really just a rough guidance.
            It's better to use `flux_gate` instead.
        flux_gate : float or function or array of floats or array of functions
            flux of ice from the left domain boundary (and tributaries)
            (unit: m3 of ice per second). If set to a high value, consider
            changing the flux_gate_buildup time. You can also provide
            a function (or an array of functions) returning the flux
            (unit: m3 of ice per second) as a function of time.
            This is overriden by `flux_gate_thickness` if provided.
        flux_gate_buildup : int
            number of years used to build up the flux gate to full value
        do_calving : bool
            switch on the chakra calving parameterisation. Ignored if not a
            tidewater glacier.
        calving_use_limiter : bool
            whether to switch on the calving limiter on the parameterisation
            makes the calving fronts thicker but the model is more stable
        calving_limiter_frac : float
            limit the front slope to a fraction of the calving front.
            "3" means 1/3. Setting it to 0 limits the slope to sea-level.
        water_level : float
            the water level. It should be zero m a.s.l, but:
            - sometimes the frontal elevation is unrealistically high (or low).
            - lake terminating glaciers
            - other uncertainties
            The default is 0. For lake terminating glaciers,
            it is inferred from PARAMS['free_board_lake_terminating'].
            The best way to set the water level for real glaciers is to use
            the same as used for the inversion (this is what
            `robust_model_run` does for you)
        apply_parameterization : func
            provide a function with arguments (model, dt) which will be called
            at the end of a normal time step. This function can take over
            the task of computing stuff such as calving. It is a simple
            way to get access to the model internals, but might not be enough
            for chakra. If not enough, just modify / adapt this code here.
        """
        super(ChakraModel, self).__init__(flowlines, mb_model=mb_model,
                                          y0=y0, glen_a=glen_a, fs=fs,
                                          is_tidewater=is_tidewater,
                                          cfl_number=cfl_number,
                                          do_kcalving=False,
                                          flux_gate_thickness=flux_gate_thickness,
                                          flux_gate=flux_gate,
                                          flux_gate_build_up=flux_gate_build_up,
                                          water_level=water_level,
                                          **kwargs)

        # Let's keep things simple for now
        if len(self.fls) > 1:
            raise InvalidParamsError('Chakra wants only one flowline '
                                     'for now.')

        # Here some chakra specific parameters
        self.do_calving = do_calving

        # When simple param switched on this will updated
        self.apply_parameterization = apply_parameterization

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # Simple container
        mbs = []

        # Loop over tributaries to determine the flux rate
        for fl_id, fl in enumerate(self.fls):

            # This is possibly less efficient than zip() but much clearer
            trib = self._tributary_indices[fl_id]
            slope_stag = self.slope_stag[fl_id]
            thick_stag = self.thick_stag[fl_id]
            section_stag = self.section_stag[fl_id]
            sf_stag = self.shapefac_stag[fl_id]
            flux_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            u_stag = self.u_stag[fl_id]
            flux_gate = self.flux_gate[fl_id]

            # Flowline state
            surface_h = fl.surface_h
            thick = fl.thick
            section = fl.section
            dx = fl.dx_meter

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid point
            is_trib = trib[0] is not None
            if is_trib:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])
            elif self.do_calving and self.calving_use_limiter:
                # We lower the max possible ice deformation
                # by clipping the surface slope here. It is completely
                # arbitrary but reduces ice deformation at the calving front.
                # I think that in essence, it is also partly
                # a "calving process", because this ice deformation must
                # be less at the calving front. The result is that calving
                # front "free boards" are quite high.
                # Note that clipping to water_level is arbitrary,
                # it could be any value from surface to bed (bed h being
                # the default)
                surface_h = utils.clip_min(surface_h, self.water_level)

            # Staggered gradient
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Staggered thick
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            # _fd = 2/(N+2) * self.glen_a
            N = self.glen_n
            rhogh = (self.rho * G * slope_stag) ** N
            u_stag[:] = (thick_stag ** (
                        N + 1)) * self._fd * rhogh * sf_stag ** N + \
                        (thick_stag ** (N - 1)) * self.fs * rhogh

            # Staggered section
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flux_stag[:] = u_stag * section_stag

            # Add boundary condition
            if flux_gate is not None:
                flux_stag[0] = flux_gate(self.yr)

            # CFL condition
            if not self.fixed_dt:
                maxu = np.max(np.abs(u_stag))
                if maxu > 0.:
                    cfl_dt = self.cfl_number * dx / maxu
                else:
                    cfl_dt = dt

                # Update dt only if necessary
                if cfl_dt < dt:
                    dt = cfl_dt
                    if cfl_dt < self.min_dt:
                        raise RuntimeError(
                            'CFL error: required time step smaller '
                            'than the minimum allowed: '
                            '{:.1f}s vs {:.1f}s.'.format(cfl_dt, self.min_dt))

            # Since we are in this loop, reset the tributary flux
            trib_flux[:] = 0

            # We compute MB in this loop, before mass-redistribution occurs,
            # so that MB models which rely on glacier geometry to decide things
            # (like PyGEM) can do wo with a clean glacier state
            mbs.append(self.get_mb(fl.surface_h, self.yr,
                                   fl_id=fl_id, fls=self.fls))

        # Time step
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt

        # A second loop for the mass exchange
        for fl_id, fl in enumerate(self.fls):

            flx_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            tr = self._tributary_indices[fl_id]

            dx = fl.dx_meter

            is_trib = tr[0] is not None
            # For these we had an additional grid point
            if is_trib:
                flx_stag = flx_stag[:-1]

            # Mass-balance
            widths = fl.widths_m
            mb = mbs[fl_id]
            # Allow parabolic beds to grow
            mb = dt * mb * np.where((mb > 0.) & (widths == 0), 10., widths)

            # Update section with ice flow and mass balance
            new_section = (fl.section + (
                        flx_stag[0:-1] - flx_stag[1:]) * dt / dx +
                           trib_flux * dt / dx + mb)

            # Keep positive values only and store
            fl.section = utils.clip_min(new_section, 0)

            # Add the last flux to the tributary
            # this works because the lines are sorted in order
            if is_trib:
                # tr tuple: line_index, start, stop, gaussian_kernel
                self.trib_flux[tr[0]][tr[1]:tr[2]] += \
                    utils.clip_min(flx_stag[-1], 0) * tr[3]

            # If we use a flux-gate, store the total volume that came in
            self.flux_gate_m3_since_y0 += flx_stag[0] * dt

            # --- The space below is for calving only ---
            # CHAKRA: ADD CODE HERE
            # from this point onwards all ice deformation processes have
            # been applied and the model is ready for the next time step.
            # Something calving related could be coded here, or the other
            # possibility is to use the "apply parameterisation"
            # mechanism which is basically equivalent since it gives access
            # to all object internals. In short, it is a matter of taste,
            # but it also depends on how much chakra will mess around with
            # the numerics, in which case coding here and messing around
            # might be the better way to move forward.

            # Apply some parameterization?
            if self.apply_parameterization is not None:
                # we give the object and time step as params
                self.apply_parameterization(self, dt)

        # Next step
        self.t += dt
        return dt
