# internal_photoev.py
#
# Author: Alice Somigliana
# Date : June 9th, 2022
#
# Computes the mass loss term in the MHD scenario. Based on Giovanni Rosotti's Spock code. 
################################################################################
from __future__ import print_function

import numpy as np
from scipy import integrate
from .constants import *
from .disc import AccretionDisc

class internal_photoev():
    """
    Compute the photoevaporative mass loss term following Owen et al. (2012)
    """

    def __init__(self, disc):

        self._grid = disc.grid
        self._radius = self._grid.Rc
        self._sigma = disc.Sigma

        self._floor_density = 1e-20                                # Floor density in g cm^-2

        if disc._mdot_photoev != 0:
            self.mdot_X = disc._mdot_photoev
        else:
            self.mdot_X = 6.25e-9 * (disc._star.M)**(-0.068)*(disc._L_x)**(1.14)

        self.norm_X = 1.299931298429752e-07  # Normalization factor obtained via numerical integration - \int 2 \pi x \Sigma(x) dx * au**2/Msun
        self.x = 0.85*(self._radius)*(disc._star.M)**(-1.)
        self.index_null_photoevap = np.searchsorted(self.x, 2)


        a1 = 0.15138
        b1 = -1.2182
        c1 = 3.4046
        d1 = -3.5717
        e1 = -0.32762
        f1 = 3.6064
        g1 = -2.4918

        ln10 = np.log(10.)

        self._Sigmadot_Owen_unnorm = np.zeros_like(self._radius)

        where_photoevap = self.x > 0.7

        logx = np.log10(self.x[where_photoevap])
        lnx = np.log(self.x[where_photoevap])

        x_photoev = self.x[where_photoevap]

        self._Sigmadot_Owen_unnorm[where_photoevap] = 10.**(a1*logx**6.+b1*logx**5.+c1*logx**4.+d1*logx**3.+e1*logx**2.+f1*logx+g1) * \
                            ( 6.*a1*lnx**5./(x_photoev**2.*ln10**7.) + 5.*b1*lnx**4./(x_photoev**2.*ln10**6.)+4.*c1*lnx**3./(x_photoev**2.*ln10**5.) +   
                            3.*d1*lnx**2./(x_photoev**2.*ln10**4.) + 2.*e1*lnx/(x_photoev**2. *ln10**3.) + f1/(x_photoev**2.*ln10**2.) ) * \
                            np.exp(-(x_photoev/100.)**10)


        self._Sigmadot_Owen = self._Sigmadot_Owen_unnorm/self.norm_X*self.mdot_X*(disc._star.M)**(-2)   # Normalizing
        self._Sigmadot_Owen[self._Sigmadot_Owen < 0] = 0.                                         # Setting to zero every negative value - safety 

        self.hole = False


    def Sigmadot(self, disc):
        '''
        Determine the \dot \Sigma term to add to the evolution equation.
        Check whether the gap is already opened: if not, use the standard prescription implemented in the initialization. If yes, use the switch prescription - which depends on the radius of the hole.
        '''

        sigma_threshold = 1e22                                                  # Density threshold under which the hole is considered to be open

        # Checking whether the hole is open already

        midplane_density = self._sigma/(np.sqrt(2*np.pi)*disc.H*m_H*mu_ion)
        column_density = integrate.cumtrapz(midplane_density, self._radius)
        index_hole = np.searchsorted(column_density, sigma_threshold)          # Index of the element in the density array corresponding to the opening of the gap
        self.rin_hole = self._radius[index_hole]                               # Radius of the gap

        if not self.hole and index_hole >= self.index_null_photoevap:          # Open the gap if the condition holds
                self.hole = True
                print("The hole is opened!")
                self._sigma[:index_hole] = self._floor_density

        if self.hole:

                self.y = 0.95 * (self._radius - self.rin_hole) * (disc._star.M)**(-1.) / AU
                        
                a2 = -0.438226
                b2 = -0.10658387
                c2 = 0.5699464
                d2 = 0.010732277
                e2 = -0.131809597
                f2 = -1.32285709

                self.mdot_hole_X = self.mdot_X * 0.768 * (disc._star.M)**(-0.08)              # Determining the mass-loss rate after the opening of the gap based on Owen et al. (2012) (equations B1 and B4)
                
                after_hole = self.y >0

                y_cut = self.y[after_hole]
                
                self._Sigmadot_Owen_hole = np.copy(self._Sigmadot_Owen)
                Rc_cut = self._radius[after_hole]

                self._Sigmadot_Owen_hole[after_hole] =  (a2*b2*np.exp(b2*y_cut)/Rc_cut + c2*d2*np.exp(d2*y_cut) / Rc_cut +
                    e2*f2*np.exp(f2*y_cut)/Rc_cut) * np.exp(-(y_cut/57.)**10.)
                
                norm_integral_1 = np.trapz(self._Sigmadot_Owen_hole[after_hole] * self._radius[after_hole], self._radius[after_hole])
                norm_1 = norm_integral_1 * 2 * np.pi * disc._star.M**2 / (0.95)**2

                norm_integral_2 = np.trapz(self._Sigmadot_Owen_hole[after_hole], self._radius[after_hole])
                norm_2 = norm_integral_2 * 2 * np.pi * disc._star.M * self._radius[index_hole] / 0.95 

                norm = (norm_1 + norm_2)*AU**2/Msun

                self._Sigmadot_Owen_hole[after_hole] *= - self.mdot_hole_X/norm

                return self._Sigmadot_Owen

        else:
            return self._Sigmadot_Owen

    def __call__(self, disc, dt):

        sigmadot = self.Sigmadot(disc)
        Sigma_new = disc.Sigma - dt * sigmadot

        # Check that the surface density never becomes negative

        for i in range(0, len(Sigma_new)-1):
            if Sigma_new[i] < 0:
                Sigma_new[i] = self._floor_density

        disc.Sigma[:] = Sigma_new

        disc.Sigma[0] = disc.Sigma[1]
        disc.Sigma[-1] = disc.Sigma[-2]