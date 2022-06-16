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

        grid = disc.grid

        self._floor_density = 1e-20                                # Floor density in g cm^-2

        if disc._mdot_photoev != 0:
            self.mdot_X = disc._mdot_photoev
        else:
            self.mdot_X = 6.25e-9 * (disc._star.M)**(-0.068)*(disc._L_x)**(1.14)

        self.norm_X = 1.299931298429752e-07                        # Normalization factor obtained via numerical integration - \int 2 \pi x \Sigma(x) dx * au**2/Msun
        self.x = 0.85*(grid.Rc)*(disc._star.M)**(-1.)
        self.index_null_photoevap = np.searchsorted(self.x, 2)

        a1 = 0.15138
        b1 = -1.2182
        c1 = 3.4046
        d1 = -3.5717
        e1 = -0.32762
        f1 = 3.6064
        g1 = -2.4918

        ln10 = np.log(10.)

        self._Sigmadot_Owen_unnorm = np.zeros_like(grid.Rc)

        where_photoevap = self.x > 0.7

        logx = np.log10(self.x[where_photoevap])
        lnx = np.log(self.x[where_photoevap])

        x_photoev = self.x[where_photoevap]

        self._Sigmadot_Owen_unnorm[where_photoevap] = 10.**(a1*logx**6.+b1*logx**5.+c1*logx**4.+d1*logx**3.+e1*logx**2.+f1*logx+g1) * \
                            ( 6.*a1*lnx**5./(x_photoev**2.*ln10**7.) + 5.*b1*lnx**4./(x_photoev**2.*ln10**6.)+4.*c1*lnx**3./(x_photoev**2.*ln10**5.) +   
                            3.*d1*lnx**2./(x_photoev**2.*ln10**4.) + 2.*e1*lnx/(x_photoev**2. *ln10**3.) + f1/(x_photoev**2.*ln10**2.) ) * \
                            np.exp(-(x_photoev/100.)**10)


        #THIS IS SIGMA DOT - is this CGS?
        self._Sigmadot_Owen = self._Sigmadot_Owen_unnorm/self.norm_X*self.mdot_X*(disc._star.M)**(-2)      # Normalizing
        self._Sigmadot_Owen[self._Sigmadot_Owen < 0] = 0.                                                  # Setting to zero every negative value - safety measure

        self.gap = False
        self.hole = False


    def Sigmadot(self, disc):
        return self._Sigmadot_Owen

    def __call__(self, disc, dt):
        
        grid = disc.grid
        sigma = disc.Sigma

        sigmadot = self.Sigmadot(disc)
        Sigma_new = disc.Sigma - dt * sigmadot
 
        disc.Sigma[:] = Sigma_new

        disc.Sigma[0] = disc.Sigma[1]
        disc.Sigma[-1] = disc.Sigma[-2]