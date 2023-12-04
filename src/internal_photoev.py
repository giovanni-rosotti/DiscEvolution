# internal_photoev.py
#
# Author: Alice Somigliana
# Date : June 9th, 2022
#
# Computes the mass loss term for internal photoevaporation following Owen et al. (2012). Based on Giovanni Rosotti's Spock code. 
################################################################################
from __future__ import print_function

import numpy as np
import sys
from scipy import integrate
from .constants import *
from .disc import AccretionDisc


def Sigma_dot_Picogna(R=np.array([0.]),a=0,b=0,c=0,d=0,e=0,f=0,g=0):
    # Goal: provide Sigma_dot for the ansatz of Picogna+2021
    # Input: R in au and a,b,c,... coefficients of the fits
    # Output: Sigma_dot, in au-2 since this has to be multiplied by MW(LX,MW) (eq. 5,6,7 Picogna+2021)
    ln10  = np.log(10.)
    lnR   = np.log(R)
    logR  = np.log10(R)

    Sigma_dot = ln10*(6*a*lnR**5/(R*ln10**6) +  5*b*lnR**4/(R*ln10**5)  +  \
        4*c*lnR**3/(R*ln10**4)  +  3*d*lnR**2/(R*ln10**3) +  \
        2*e*lnR/(R*ln10**2) +  f/(R*ln10) ) / \
        (2.*np.pi*R) \
        *10**(a*logR**6+b*logR**5+c*logR**4+d*logR**3+e*logR**2+f*logR+g)
    Sigma_dot[Sigma_dot<0] = 0.

    return Sigma_dot 

def MW_corr_picogna(LX_soft=1e29):
    # Goal: Eq. 6 in Ercolano+2021 to compute the MW form LX (if picogna option)
    a = -1.947e17
    b = -1.572e-4 # don't understand the relevance of this parameter: it has negligible contribution to MW_corr_picogna
    c = -2.866e-1
    d = -6.694
    #
    MW = a*np.exp( (np.log(np.log10(LX_soft) - b))**2/c ) + d
    MW = 10**MW 

    return MW

def MW_picogna(Mstar=1.,LX=1e29):
    # Goal: compile Picogna's reciepe to compute the total mass-loss rate as function of Mstar and LX
    # see Ercolano+2021
    MW_mean       = 3.93e-8*(Mstar) #mass loss-rate corresponding to the median for a give stellar mass
    L_X_soft      = 10**(0.95*np.log10(LX)+1.19) # luminosity in the 0.1-
    LX_mean       = 10**(1.54* np.log10(Mstar) + 30.31)
    L_X_soft_mean = 10**(0.95*np.log10(LX_mean)+  1.19) 
    MW = MW_mean*MW_corr_picogna(LX_soft=L_X_soft)/MW_corr_picogna(LX_soft=L_X_soft_mean)
    return MW


class internal_photoev():
    """
    Compute the photoevaporative mass loss term following the option set for the mass-loss rate profile
    """

    def __init__(self, disc, mdot_photoev_profile):

        self._grid = disc.grid
        self._radius = self._grid.Rc
        self._sigma = disc.Sigma

        self._floor_density = 1e-20                                # Floor density in g cm^-2

        self.mdot_photoev_profile = mdot_photoev_profile

        if mdot_photoev_profile=="owen":
            if disc._mdot_photoev != 0:
                self.mdot_X = disc._mdot_photoev
            else:
                self.mdot_X = 6.25e-9 * (disc._star.M)**(-0.068)*(disc._L_x)**(1.14)

            self.norm_X = 1.299931298429752e-07  # Normalization factor obtained via numerical integration - \int 2 \pi x \Sigma(x) dx * au**2/Msun
            self.x      = 0.85*(self._radius)*(disc._star.M)**(-1.)
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
                                ( 6.*a1*lnx**5./(x_photoev**2.*ln10**7.) + 5.*b1*lnx**4./(x_photoev**2.*ln10**6.)+4.*c1*lnx**3./(x_photoev**2.*ln10**5.) + \
                                3.*d1*lnx**2./(x_photoev**2.*ln10**4.) + 2.*e1*lnx/(x_photoev**2. *ln10**3.) + f1/(x_photoev**2.*ln10**2.) ) * \
                                np.exp(-(x_photoev/100.)**10)


            self._Sigmadot_Owen = self._Sigmadot_Owen_unnorm/self.norm_X*self.mdot_X*(disc._star.M)**(-2)   # Normalizing
            self._Sigmadot_Owen[self._Sigmadot_Owen < 0] = 0.                                         # Setting to zero every negative value - safety 
            self._Sigmadot_Owen = self._Sigmadot_Owen/(2.*np.pi) # code units !

            self.hole = False

        elif mdot_photoev_profile=="picogna":
            if disc._mdot_photoev != 0:
                self.mdot_X = disc._mdot_photoev
            else:
                self.mdot_X = MW_picogna(Mstar=disc._star.M,LX=disc._L_x)

            mstars=np.array([0.1,0.3,0.5,1.0])
            a=np.array([ -3.8337,  -1.3206,  -1.2320, -0.6344])
            b=np.array([ 22.91,    13.0475,  10.8505,  6.3587])
            c=np.array([-55.1282, -53.6990, -38.6939,-26.1445])
            d=np.array([ 67.8919, 117.6027,  71.2489, 56.4477])
            e=np.array([-45.0138,-144.3769, -71.4279,-67.7403])
            f=np.array([ 16.2977,  94.7854,  37.8707, 43.9212])
            g=np.array([ -3.5426, -26.7363,  -9.3508,-13.2316])

            mstar = disc._star.M

            x=self._radius


            ind  = np.searchsorted(mstars,mstar,side='left')
            if (ind == 0):             # mstar smaller than 0.1Msun => no extrapolations !
                Sigma_dot       = Sigma_dot_Picogna(R=x,a=a[0],b=b[0],c=c[0],d=d[0],e=e[0],f=f[0],g=g[0])
            elif (ind == len(mstars)): # mstar larger than 1Msun => no exptrapolations !
                Sigma_dot       = Sigma_dot_Picogna(R=x,a=a[-1],b=b[-1],c=c[-1],d=d[-1],e=e[-1],f=f[-1],g=g[-1])
            else:                      # interpolation
                Sigma_dot_left  = Sigma_dot_Picogna(R=x,a=a[ind-1], b=b[ind-1], c=c[ind-1],d=d[ind-1], e=e[ind-1],f=f[ind-1], g=g[ind-1])
                Sigma_dot_right = Sigma_dot_Picogna(R=x,a=a[ind],   b=b[ind],   c=c[ind],  d=d[ind],   e=e[ind],  f=f[ind],   g=g[ind])
                Sigma_dot       = Sigma_dot_left + (mstar - mstars[ind-1]) * (Sigma_dot_right-Sigma_dot_left)/(mstars[ind] - mstars[ind-1] )

            self._Sigmadot_Picogna =  3.93e-8*mstar*Msun*Sigma_dot/AU**2/yr    # in g s-1 cm-2 => scalings are not necessary but just to check that we recover MW in Picogna+2021 fig. 6
            norm = np.trapz(2*np.pi*x*AU*self._Sigmadot_Picogna,x*AU)
            self._Sigmadot_Picogna *= self.mdot_X*Msun/yr/norm                 # in g s-1 cm-2 with correct resulting MW
            self._Sigmadot_Picogna *= yr/(2.*np.pi)                            # in code units
 

        elif mdot_photoev_profile=="komaki":
            if disc._mdot_photoev != 0:
                self.mdot_X = disc._mdot_photoev
            else:
                raise NotImplementedError("The requested photo-evaporation profile type " + mdot_photoev_profile + " can't use LX to compute mass-loss rate. Please set lx=0 and prescribe a value for mdot_photoev")

            mstar = disc._star.M
            rad   = self._radius
            x     = self._radius/(8.87*mstar) # radius normalised by the grav radius rg (Komaki+21)
            logx = np.log10(x)

            self._Sigmadot_Komaki = 0.693*logx**5 -0.95*logx**4 -0.038*logx**3 +0.678*logx**2 -1.67*logx -12.6
            self._Sigmadot_Komaki = 10**self._Sigmadot_Komaki * np.exp(-(x/17.)**2) # B. Tabone: I added a tapper to avoid divergence at large distance


            #normalise profile
            norm = np.trapz(2*np.pi*rad*AU*self._Sigmadot_Komaki,rad*AU)
            self._Sigmadot_Komaki *= self.mdot_X*Msun/yr/norm # in g cm-2 s-1

            self._Sigmadot_Komaki *= yr/(2.*np.pi)            # in code units


        elif mdot_photoev_profile=="alexander":
            print("You selected alexander mass-loss rate: this option is not validated. Please test this option (units for example)!")
            alpha_b  = 2.6e-13  # recombination coefficient of atomic hydrogen
            self.sigcrit=1e-5
            c_s=10 * 1000 * 100. #10 km/s in cgs units
            G_cgs = 6.67e-8
            mstar_cgs = disc._star.M*Msun
            r_g_cgs = G_cgs * mstar_cgs / c_s**2
            self.r_g = r_g_cgs/AU #gravitational radius in AU
            phi_phlux=1e42
            C_1=0.14
            n_g = C_1 *(3*phi_phlux/ (4*np.pi*alpha_b * r_g_cgs**3) )**0.5
            A_photo=0.3423
            B_photo=-0.3612
            D_photo=0.2457
            self.a_direct=2.42
            C_2=0.235
            self.const_direct=2*C_2*mu_ion*c_s*m_p_cgs*(phi_phlux/(4*np.pi*alpha_b))**(1./2)
            xph=self._radius/self.r_g
            n_0=n_g*(2/(xph**(15./2) + xph**(25./2) ))**(1./5)
            u_l=np.zeros((self._radius.size))
            rgreat=np.where(self._radius>0.1*self.r_g)
            u_l[rgreat]=c_s*A_photo*np.exp(B_photo* (xph[rgreat]-0.1))*(xph[rgreat] -0.1)**D_photo
            self._Sigmadot_Alexander=2*n_0*u_l*mu_ion*m_p_cgs #"Diffuse" profile - see appendix of Alexander & Armitage (2007)
            self._Sigmadot_Alexander *= disc._mdot_photoev * Msun / yr / np.trapz(2*np.pi*self._radius*AU*self._Sigmadot_Alexander,self._radius*AU)

            self.directprecomputed =  self.const_direct*(disc._eos.H/self._radius)**(-1./2)*(self._radius)**(-self.a_direct)
            self.sigthin = m_p_cgs / sig_h_atomar
            chi = 2.5
            invexpchi2 = 1./np.exp(-chi**2/2.)
            self.constantdensity = np.sqrt(2*np.pi)*m_p_cgs*mu_ion*invexpchi2
            self.ir_01 = np.searchsorted(self._radius,0.1)
            self.ir_1 = np.searchsorted(self._radius,1.)


        else:
             raise NotImplementedError("The requested photo-evaporation profile type " + mdot_photoev_profile + " is not implemented yet")


    def Sigmadot(self, disc):
        '''
        Determine the \dot \Sigma term to add to the evolution equation.
        Check whether the gap is already opened: if not, use the standard prescription implemented in the initialization. If yes, use the switch prescription - which depends on the radius of the hole.
        '''
        self._flag_dispersion = False
        if self.mdot_photoev_profile=="owen":
            sigma_threshold = 1e22                                                  # Density threshold under which the hole is considered to be open
            self._flag_dispersion = False

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

                    #if not any(after_hole):
                    if np.sum(after_hole)<2:
                        self._flag_dispersion = True
                        print('The hole is too large now - I will stop the evolution')
                        return 0, True

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

                    return self._Sigmadot_Owen, False

            else:

                return self._Sigmadot_Owen, False
            
        elif self.mdot_photoev_profile=="picogna":
            return self._Sigmadot_Picogna, False

        elif self.mdot_photoev_profile=="komaki":
            return self._Sigmadot_Komaki, False
        
        elif self.mdot_photoev_profile=="alexander":
            self._flag_dispersion = False
            isdirect= (disc.Sigma[self.ir_01] < self.sigcrit and disc.Sigma[self.ir_1] < self.sigcrit)
            if isdirect:
                #raise NotImplementedError()
                dens_midplane=disc.Sigma/(np.sqrt(2*np.pi)*disc._eos.H*AU*m_p_cgs*mu_ion)
                tau=integrate.cumtrapz(dens_midplane,self._radius*AU)*sig_h_atomar
                self.rin2=self._radius[ max(np.searchsorted(tau, 4.61)-1,0) ]
                Sigmadot_direct = self.directprecomputed*self.rin2**(-3./2)/ (self.rin2)**(-self.a_direct)
                return self._directprofile(disc), False

            else:
                return self._Sigmadot_Alexander, False
        else:
             raise NotImplementedError("The requested photo-evaporation profile type " + self.mdot_photoev_profile + " is not implemented yet")

    def __call__(self, disc, dt):

        sigmadot, flag = self.Sigmadot(disc)

        if flag:
            return True

        Sigma_new = disc.Sigma - dt * sigmadot

        # Check that the surface density never becomes negative

        Sigma_new[Sigma_new < 0] = self._floor_density

        disc.Sigma[:] = Sigma_new

    @staticmethod
    def return_flag_dispersion(self):
        return self._flag_dispersion