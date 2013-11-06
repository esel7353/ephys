#!/usr/bin/python3

import numpy as np
import scipy.constants as sc

# all SI units are unity
J = 1
V = 1

# conversion to SI units
eV = sc.e * V
MeV = sc.mega * eV

def bindingEnergy(A, Z):
  """
  The binding energy of a nucleus. The formula is known as the 'Bethe Weizsäcker mass
  formula'. The binding energy is defined as the energy which is freed when the nucleus
  is formed from independent nucleons. Therefore the binding energy of seperate
  nucleons is null.
  Used constants might be subject to change.

  Parameter:
    A - number of nucleons, i.e number of protons and neutrons
    Z - number of protons
    The parameters might be numpy arrays or scalars. If both are numpy arrays, they
    are required to have the same shape.

  Return:
    binding Energy in Joule

  References: [1]
  """
  # number of neutrons
  N = A - Z 

  # empirical constants [1]
  a_V = 15.84 * MeV   # volume term
  a_S = 18.33 * MeV   # surface term
  a_F = 23.2  * MeV   # asymmetry term
  a_C = 0.714 * MeV   # Coulomb term
  a_P = 12    * MeV   # pairing term

  # even-odd nuclei
  Zmod = Z % 2
  Nmod = N % 2

  ee = (Zmod == 0) * (Nmod == 0) # int or numpy index array 
  oo = (Zmod == 1) * (Nmod == 1) # int or numpy index array 
  delta = ee - oo

  return a_V * A - a_S * np.power(A, 2/3) - a_F * np.square(N-Z) / A - a_C * np.square(Z) * np.power(A, -1/3) + delta * a_P / np.sqrt(A)
  

if __name__ == "__main__":
  import pylab

  A = np.array(range(3, 250))

  Z = np.round(A / 2.5)

  pylab.plot(A, bindingEnergy(A,Z)/A/MeV)
  pylab.xlabel("Massenzahl $A$")
  pylab.ylabel("Bindungsernergie pro Nukleon $E/A \\, / \\, \mathrm{MeV}$")


  pylab.savefig("bindingEnergy.png", dpi=300)


