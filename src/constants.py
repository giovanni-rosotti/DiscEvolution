"""
Useful constants

	tiny : standard small number to avoid division by zero

	G: Gravitational constant G - set to 1
	Omega0 : Keplerian angular velocity with M = Msun and R = 1 AU [s^(-1)]
	Msun: Solar mass in cgs [g]
	Rsun : Solar radius in cgs [cm]
	Mearth : Earth mass in cgs [g]
	AU : Astronomic unit in cgs [cm]
	yr : Year in seconds

	k_B : Boltzmann K-constant in cgs [erg/K]
	sig_SB : Stefan-Boltzmann sigma-constant in cgs [erg/(cm^2 s K^4)]
	m_H : Hydrogen (proton) mass in cgs [g]
	sig_H2 : Cross-section of H2 molecule [cm^2]
	GasConst : = k_B/m_H [cm^2/(K s^2)]
"""

tiny = 1e-100

G = 1.
Omega0 = 1.99102e-7
Msun = 1.989e33
Rsun = 6.96e10
Mearth = 5.972e27
AU = 1.496e13
yr = 3.1536e7

k_B = 1.3806e-16
sig_SB = 5.6704e-5
m_H = 1.6737e-24
sig_H2 = 2e-15
GasConst = 8.314472e7
mu_ion = 1.35