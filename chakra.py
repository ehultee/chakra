"""Chakra: smooth physically based calving in OGGM.

Authors: Lizz Ultee, Fabien Maussion & the OGGM developers
"""

# Builtins
import warnings

# External libs
import numpy as np
import pandas as pd
from scipy import optimize

# Locals
import oggm.cfg as cfg
from oggm.exceptions import InvalidParamsError
from oggm.core.massbalance import LinearMassBalance, MassBalanceModel

# Constants
from oggm.cfg import SEC_IN_HOUR, SEC_IN_DAY, SEC_IN_YEAR
from oggm.cfg import G
from oggm.core.flowline import FlowlineModel, RectangularBedFlowline
from oggm.core.inversion import sia_thickness


def find_sia_flux_from_thickness(slope, width, thick, glen_a=None, fs=None):
    """Find the ice flux produced by a given thickness and slope.

    This can be done analytically but I'm too lazy and use OGGM instead.
    """
    def to_minimize(x):
        h = sia_thickness(slope, width, x[0],  glen_a=glen_a, fs=fs)
        return (thick - h)**2
    out = optimize.minimize(to_minimize, [1], bounds=((0, 1e12),))
    flux = out['x'][0]

    # Sanity check
    minimum = to_minimize([flux])
    if minimum > 1:
        warnings.warn('We did not find a proper flux for this thickness',
                      RuntimeWarning)
    return flux


class CalvingModel(FlowlineModel):
    """A fork/copy of the flowline model used by OGGM in production.

    We removed the shape factors and added a flux-gate option.

    It solves for the SIA along the flowline(s) using a staggered grid. It
    computes the *ice flux* between grid points and transports the mass
    accordingly (also between flowlines).

    This model is numerically less stable than fancier schemes, but it
    is fast and works with multiple flowlines of any bed shape (rectangular,
    parabolic, trapeze, and any combination of them).

    We test that it conserves mass in most cases, but not on very stiff cliffs.
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., inplace=False, fixed_dt=None, cfl_number=0.05,
                 min_dt=1*SEC_IN_HOUR, max_dt=10*SEC_IN_DAY,
                 time_stepping='user', flux_gate_thickness=None,
                 flux_gate=None, flux_gate_build_up=500, **kwargs):
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
            for adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx / max_u).
            Schoolbook theory says that the scheme is stable
            with CFL=1, but practice does not. There is no "best" CFL number:
            small values are more robust but also slowier...
        min_dt : float
            with high velocities, time steps can become very small and your
            model might run very slowly. In production we just take the risk
            of becoming unstable and prevent very small time steps.
        max_dt : float
            just to make sure that the adaptive time step is not going to
            choose too high values either. We could make this higher I think
        time_stepping : str
            let OGGM choose default values for the parameters above for you.
            Possible settings are: 'ambitious', 'default', 'conservative',
            'ultra-conservative'.
        max_dt : float
            just to make sure that the adaptive time step is not going to
            choose too high values either. We could make this higher I think
        flux_gate_thickness : float
            flux of ice from the left domain boundary. Units of m of ice
            thickness. If set to a high value, the model will slowly build it
            up in order not to force the model too much. Note that unrealistic
            values won't be met by the model, so this is really just a rough
            help.
        flux_gate : float
            a flux value (unit: m3 of ice per year). Will be overriden by
            flux_gate_thickness if provided.
        flux_gate_buildup : int
            number of years until the flux is reached
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass-balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        """
        super(CalvingModel, self).__init__(flowlines, mb_model=mb_model,
                                           y0=y0, glen_a=glen_a, fs=fs,
                                           inplace=inplace, **kwargs)

        if time_stepping == 'ambitious':
            cfl_number = 0.1
            min_dt = 1*SEC_IN_DAY
            max_dt = 15*SEC_IN_DAY
        elif time_stepping == 'default':
            cfl_number = 0.05
            min_dt = 1*SEC_IN_HOUR
            max_dt = 10*SEC_IN_DAY
        elif time_stepping == 'conservative':
            cfl_number = 0.01
            min_dt = SEC_IN_HOUR
            max_dt = 5*SEC_IN_DAY
        elif time_stepping == 'ultra-conservative':
            cfl_number = 0.01
            min_dt = SEC_IN_HOUR / 10
            max_dt = 5*SEC_IN_DAY
        else:
            if time_stepping != 'user':
                raise ValueError('time_stepping not understood.')

        self.dt_warning = False
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.cfl_number = cfl_number

        # When switched on
        self.simple_calving_m3_since_y0 = 0.  # total calving since time y0

        # Flux gate
        self.flux_gate = flux_gate
        self.flux_gate_yr = flux_gate_build_up
        if flux_gate_thickness is not None:
            # Compute the theoretical ice flux from the slope at the top
            fl = self.fls[-1]
            slope = (fl.surface_h[0] - fl.surface_h[1]) / fl.dx_meter
            if slope == 0:
                raise ValueError('I need a slope to compute the flux')
            self.flux_gate = find_sia_flux_from_thickness(slope,
                                                          fl.widths_m[0],
                                                          flux_gate_thickness,
                                                          glen_a=self.glen_a,
                                                          fs=self.fs)

        # Memory optimisation and diagnostics
        self.slope_stag = []
        self.thick_stag = []
        self.section_stag = []
        self.u_stag = []
        self.flux_stag = []
        self.trib_flux = []
        for fl, trib in zip(self.fls, self._tributary_indices):
            nx = fl.nx
            # This is not staggered
            self.trib_flux.append(np.zeros(nx))
            # We add an additional fake grid point at the end of these
            if trib[0] is not None:
                nx = fl.nx + 1
            # +1 is for the staggered grid
            self.slope_stag.append(np.zeros(nx+1))
            self.thick_stag.append(np.zeros(nx+1))
            self.section_stag.append(np.zeros(nx+1))
            self.u_stag.append(np.zeros(nx+1))
            self.flux_stag.append(np.zeros(nx+1))

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # This is to guarantee a precise arrival on a specific date if asked
        min_dt = dt if dt < self.min_dt else self.min_dt

        # Loop over tributaries to determine the flux rate
        for fl_id, fl in enumerate(self.fls):

            # This is possibly less efficient than zip() but much clearer
            trib = self._tributary_indices[fl_id]
            slope_stag = self.slope_stag[fl_id]
            thick_stag = self.thick_stag[fl_id]
            section_stag = self.section_stag[fl_id]
            flux_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            u_stag = self.u_stag[fl_id]

            # Flowline state
            surface_h = fl.surface_h
            thick = fl.thick
            section = fl.section
            dx = fl.dx_meter

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid point
            if trib[0] is not None:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])

            # Staggered gradient
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]

            # Convert to angle?
            # slope_stag = np.sin(np.arctan(slope_stag))

            # Staggered thick
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]

            # Staggered velocity (Deformation + Sliding)
            # _fd = 2/(N+2) * self.glen_a
            N = self.glen_n
            rhogh = (self.rho*G*slope_stag)**N
            u_stag[:] = (thick_stag**(N+1)) * self._fd * rhogh + \
                        (thick_stag**(N-1)) * self.fs * rhogh

            # Staggered section
            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
            section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flux_stag[:] = u_stag * section_stag

            # Add boundary condition
            if self.flux_gate is not None:
                fac = 1 - (self.flux_gate_yr - self.yr) / self.flux_gate_yr
                flux_stag[0] = self.flux_gate * np.clip(fac, 1e-4, 1)

            # CFL condition
            maxu = np.max(np.abs(u_stag))
            if maxu > 0.:
                _dt = self.cfl_number * dx / maxu
            else:
                _dt = self.max_dt
            if _dt < dt:
                dt = _dt

            # Since we are in this loop, reset the tributary flux
            trib_flux[:] = 0

        # Time step
        self.dt_warning = dt < min_dt
        dt = np.clip(dt, min_dt, self.max_dt)

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

            # Mass balance
            widths = fl.widths_m
            mb = self.get_mb(fl.surface_h, self.yr, fl_id=fl_id)
            # Allow parabolic beds to grow
            mb = dt * mb * np.where((mb > 0.) & (widths == 0), 10., widths)

            # Update section with ice flow and mass balance
            new_section = (fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt/dx +
                           trib_flux*dt/dx + mb)

            # Keep positive values only and store
            fl.section = new_section.clip(0)

            # Add the last flux to the tributary
            # this works because the lines are sorted in order
            if is_trib:
                # tr tuple: line_index, start, stop, gaussian_kernel
                self.trib_flux[tr[0]][tr[1]:tr[2]] += (flx_stag[-1].clip(0) *
                                                       tr[3])

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
            - ice_flux, tributary_flux: m3 of *ice* per year
            - u: m per year
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
        df['slope'] = (var[1:nx+1] + var[:nx])/2
        var = self.flux_stag[fl_id]
        df['ice_flux'] = (var[1:nx+1] + var[:nx])/2 * cfg.SEC_IN_YEAR
        var = self.u_stag[fl_id]
        df['u'] = (var[1:nx+1] + var[:nx])/2 * cfg.SEC_IN_YEAR

        # Not Staggered
        df['tributary_flux'] = self.trib_flux[fl_id] * cfg.SEC_IN_YEAR

        return df


class FixedMassBalance(MassBalanceModel):
    """Constant mass-balance, everywhere."""

    def __init__(self, mb=0.):
        """ Initialize.

        Parameters
        ----------
        mb: float
            Fix the mass balance to a certain value (unit: [mm w.e. yr-1])

        """
        super(FixedMassBalance, self).__init__()
        self.hemisphere = 'nh'
        self.valid_bounds = [-2e4, 2e4]  # in m
        self._mb = mb

    def get_monthly_mb(self, heights, year=None, fl_id=None):
        mb = np.asarray(heights) * 0 + self._mb
        return mb / SEC_IN_YEAR / self.rho

    def get_annual_mb(self, heights, year=None, fl_id=None):
        mb = np.asarray(heights) * 0 + self._mb
        return mb / SEC_IN_YEAR / self.rho


class WaterMassBalance(LinearMassBalance):
    """MM as a linear function of altitude, and a fixed value underwater.
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

    def get_monthly_mb(self, heights, year=None, fl_id=None):
        heights = np.asarray(heights)
        mb = super(WaterMassBalance, self).get_monthly_mb(heights)
        return np.where(heights < 0, self.underwater_melt, mb)

    def get_annual_mb(self, heights, year=None, fl_id=None):
        heights = np.asarray(heights)
        mb = super(WaterMassBalance, self).get_annual_mb(heights)
        return np.where(heights < 0, self.underwater_melt, mb)


def dummy_tidewater_bed(gridsize=200, gridlength=6e4, widths_m=600,
                        b_0=260, alpha=0.017, b_1=350, x_0=4e4, sigma=1e4):

    dx_meter = gridlength / gridsize
    x = np.arange(gridsize+1) * dx_meter
    bed_h = b_0 - alpha * x + b_1 * np.exp(-((x - x_0) / sigma)** 2)
    surface_h = bed_h
    widths = surface_h * 0. + widths_m / dx_meter
    return [RectangularBedFlowline(dx=1, map_dx=dx_meter, surface_h=surface_h,
                                   bed_h=bed_h, widths=widths)]
