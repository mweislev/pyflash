# -*- coding: utf-8 -*-
import numpy as np

__author__ = "Michael Weis"
__version__ = "0.0.1.0"

cm = 1.
sec = 1.
gram = 1.
rad = 1.
k_b = 1.380649e-16    # Boltzmann constant (erg/K)
G_g = 6.67430e-8    # Gravitational constant (cgs)
M_sol = 1.9885e+33 *gram        # solar mass (g)
Myr = 3.15576e+13 *sec          # 1e+6 julian years
kyr = 1e-3*Myr                  # 1e+3 julian years
year = 1e-6*Myr                 # 1e+0 julian years
day = 24.*60.*60. *sec
week = 7.*day
m_a = 1.66053906660e-24 *gram   # unified atomic mass unit Dalton
m_e = 9.1093837015e-28 *gram    # electron rest mass
au = 1.495978707e+13 *cm        # astronomical unit (IAU 2012-B2)
pc = 648000./np.pi *au          # parsec (IAU 2015-B2)
km = 1e+5 *cm                   # Kilometer
deg = np.pi/180.

L_sol = 3.9e33                  # solar luminosity (cgs)

