# mhd_massloss.py
#
# Author: Alice Somigliana
# Date : March 4th, 2022
#
# Computes the mass loss term in the MHD scenario.
################################################################################
from __future__ import print_function

import numpy as np
from .constants import *
from .disc import AccretionDisc

class MHD_massloss():
    """
    Compute the mass loss term in the MHD disc winds + viscosity evolution equation (Tabone et al. 2021).

    args:

        alpha_DW:   alpha disc wind parameter (Tabone et al. 2021)
        leverarm:   lambda parameter (Tabone et al. 2021)
        xi:         xi parameter (Tabone et al. 2021)
    """

    def __init__(self, alpha_DW, leverarm, xi):

        self._alpha_DW = alpha_DW
        self._leverarm = leverarm
        self._xi = xi

    def Sigmadot(self, grid, disc, sigma):
        sigmadot = (3*self._alpha_DW*sigma*disc._eos._f_cs(grid.Rc)**2)/(4*(self._leverarm-1)*grid.Rc**2*disc.new_Omega_k(grid.Rc))
        return sigmadot

    def __call__(self, disc, dt):
        
        grid = disc.grid
        sigma = disc.Sigma

        sigmadot = self.Sigmadot(grid, disc, sigma)
        Sigma_new = disc.Sigma - dt * sigmadot
        
        disc.Sigma[:] = Sigma_new

        disc.Sigma[0] = disc.Sigma[1]*(grid.Rc[1]/grid.Rc[0])**(1-self._xi)
        disc.Sigma[-1] = disc.Sigma[-2]*(grid.Rc[-2]/grid.Rc[-1])**(1-self._xi)
