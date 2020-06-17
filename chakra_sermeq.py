#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chakra_sermeq: a version of SERMeQ calving for implementation with OGGM

Created on Mon Jun 15 13:55:02 2020

@author: EHU
"""

import numpy as np
import math
from scipy import interpolate

## Global constants
H0 = 1e3  # characteristic height for nondimensionalisation 
L0 = 10e3 # characteristic length (10km)
G = 9.8 # acceleration due to gravity in m/s^2
RHO_ICE = 920.0 #ice density kg/m^3
RHO_SEA = 1020.0 #seawater density kg/m^3


class PlasticGlacier(object):
    """A calving glacier with shape defined by plastic yielding at terminus.
    """
    def __init__(self, yield_strength=150e3):
        """Initialize the glacier

        Parameters
        ----------
        yield_strength : float, optional
            Yield strength of ice in Pa. The default is 150e3.

        """
        self.yield_strength = yield_strength
    
    def set_bed_function(self, x, bed_vals):
        """Set up a continuous function of x describing subglacial topography

        Parameters
        ----------
        x : array
            Positions x (m/L0) along the flowline.
        bed_vals : array
            Bed elevation (m/H0) at each position x.
        """
        bf=interpolate.interp1d(x, bed_vals, 'linear')
        self.bed_function = bf
    
    def bingham_const(self, bed_elev=None, thick=None):
        """Functional form of constant Bingham number.
        Bingham number is nondimensional yield stress.

        Parameters
        ----------
        Don't set. All necessary inherited.

        Returns
        -------
        B : float 
        """
        return self.yield_strength / (RHO_ICE*G*H0**2 /L0)
    
    def bingham_var(self, bed_elev, thick, mu=0.01):
        """A spatially variable Bingham number
        Adjusts according to a Mohr-Coulomb basal yield condition.
        
        Parameters
        ----------
        bed_elev : float
            Bed elevation, nondimensional (m/H0).
        thick : float
            Ice thickness, nondimensional (m/H0).
        mu : float, optional
            Cohesion, a coefficient between 0 and 1. Default is 0.01.

        Returns
        -------
        B : float
        """
        if bed_elev<0:
            D = -bed_elev #Water depth D the nondim bed topography value when Z<0
        else:
            D = 0
        N = RHO_ICE*G*H0*thick - RHO_SEA*G*D*H0 #Normal stress at bed
        tau_y = self.yield_strength + mu*N
        return tau_y/(RHO_ICE*G*H0**2/L0) 

    def balance_thickness(self, bed_elev,B):
        """Water balance ice thickness

        Parameters
        ----------
        bed_elev : float
            Bed elevation, nondimensional (m/H0).
        B : float
            Bingham number at this point.

        Returns
        ------
        h : float
            Nondimensional ice thickness satisfying water balance condition.

        """
        if bed_elev<0:
            D = -1*bed_elev
        else:
            D = 0
        return (2*B*H0/L0) + math.sqrt((RHO_SEA*(D**2)/RHO_ICE)+(H0*B/L0)**2)
    
    def flotation_thickness(self, bed):
        """Minimum ice thickness before flotation.
        Argument:
            bed is nondim bed value at point of interest
        """
        if bed<0:
            D = -bed
        else:
            D = 0
        return (RHO_SEA/RHO_ICE)*D

    def plastic_profile(self, Bfunction, startpoint, hinit, endpoint, npoints=1000):
        """Make a plastic glacier surface profile over given bed topography.
        

        Parameters
        ----------
        Bfunction : function (1D)
            Nondimensional yield function; bingham_const or bingham_var.
        startpoint : float
            Point along flowline to start integration, in m/L0.
        hinit : float
            Ice surface elevation at startpoint
        endpoint : float
            Point along flowline to stop integration, in m/L0.
        npoints : float
            Number of model points to use. Default is 1000.

        Returns
        ------
        list of:
            x_vals : array
                x positions (m/L0) of each modelled point.
            SEarr : array
                Glacier surface elevation (m/H0) at each modelled point.
            basearr : array
                Bed elevation (m/H0) at each point.

        """
        horiz = np.linspace(startpoint, endpoint, npoints)
        dx = np.mean(np.diff(horiz))
    
        if dx>0:
            print('Detected: running from upglacier down to terminus.')
        elif dx<0:
            print('Detected: running from terminus upstream.')
        
        SEarr = []
        thickarr = []
        basearr = []
        
        SEarr.append(hinit)
        thickarr.append(hinit-self.bed_function(startpoint))
        basearr.append(self.bed_function(startpoint))
        for x in horiz[1::]:
            bed = self.bed_function(x)  # value of interpolated bed function
            modelthick = thickarr[-1]
            B = Bfunction(bed, modelthick) # Bingham number at this position
            #Break statements for thinning below yield, water balance, or flotation
            if dx>0:
                if modelthick<self.balance_thickness(bed,B):
                    print('Thinned below water balance at x={} km'.format(10*x))
                    break
            if modelthick<self.flotation_thickness(bed):
                print('Thinned below flotation at x= {} km'.format(10*x))
                break
            if modelthick<4*B*H0/L0:
                print('Thinned below yield at x={} km'.format(10*x))
                break
            else:
                basearr.append(bed)
                SEarr.append(SEarr[-1]-(B/modelthick)*dx) 
                thickarr.append(SEarr[-1]-basearr[-1])
        
        return (horiz[0:len(SEarr)], SEarr, basearr)

def glen_u(width, A=3.0e-25, basal_yield=150e3, surface_slope=0.2, thickness=500):
    """Ideal Glen's-law mean velocity for a laterally confined glacier.

    Parameters
    ----------
    width : float
        Glacier width (m).
    A : float, optional
        Glen's A parameter (Pa s-1/3). Default is 3.0e-25.
    basal_yield : float, optional
        Basal yield strength (Pa), usually =ice strength.  Default is 150e3.
    surface_slope : float, optional
        Glacier surface slope (m/m).  Default is 0.2.
    thickness : float, optional
        Ice thickness (m).  Default is 500.

    Returns
    -------
    Mean velocity (m s-1).
    """
    ## Centerline velocity u0:
    u = A/2 * ((RHO_ICE * G * surface_slope - (basal_yield/thickness))**3) * (width/2)**4
    
    return 0.6*u