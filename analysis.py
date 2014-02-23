#!/usr/bin/python3
################################################################################
#
# Copyright (C) 2013, Frank Sauerburger
#   published under MIT license (see below)
#
################################################################################
#
#  Experimental Data Analysis Tool
#
################################################################################
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

import numpy as np
import math
import scipy.constants as sc
import collections
from data import Quantity, Meter

def mean(it, stddev=None):
  """
  Calculates the (weighted) mean, and if standard deviations are give, also the
  standard deviation of the mean.

  Parameters:
    it     - This must be an iterable containing the values to be averaged.
             This can also be Quantity object, which also contains standard
             deviations. In this case the stddev parameter must be None.

    stddev - This can be an iterable or a single value or None. If this is a
             single value, all values are weighted equally, if this is an
             iterable the weighted mean is calculated instead. If this is
             None, all values are weighted equally and no standard deviation
             of th mean is caluculated.
             If it is a Quantity object, this must be None.
  Return:
    The return value depends on the given parameters.

    No stddev, None:      mean
    Single valued stddev: (mean, stddev of mean)
    Iterable stddev:      (mean, stddev of mean)
    Quantity object:      Quantity(mean, stddev of mean, Unit) 

  """

  if isinstance(it, np.ndarray):
    if isinstance(stddev, np.ndarray):
      if (stddev==0).any():
        return mean(it[stddev==0])
      else:
        weights = 1 / stddev**2
        m = sum(weights * it) / sum(weights)
        stddev  = np.sqrt(1 / sum(weights))
        return (m, stddev)
    elif isinstance(stddev, collections.Iterable):
      return mean(it, np.array(stddev))
    elif stddev is None:
      return mean(it, 0)[0]
    else:
      m = sum(it) / len(it)
      stddev = stddev / math.sqrt(len(it))
      return (m, stddev)
  elif isinstance(it, Quantity):
    if stddev is not None: raise ValueError('If Quantity given std must not be give')
    (m, stddev) = mean(it.value, it.variance)
    return Quantity(m, stddev, it.uvec)
  elif isinstance(it, collections.Iterable):
    if stddev is None:
      return mean(np.array(it), 0)[0]
    else:
      return mean(np.array(it), stddev)
  else:
    raise TypeError('First argument must be iterable.')



if __name__ == '__main__':
  x = np.arange(10)
  s = np.zeros(10) + 0.1
  q = Quantity(x, 0, Meter) | 0.1
  r = Quantity(x, s, Meter)

  assert mean(x, 0.1)  == (4.5, 0.1 / math.sqrt(10))
  assert mean(x, s)    == (4.5, 0.1 / math.sqrt(10))
  assert mean(q)       == Quantity(4.5, 0.1 / math.sqrt(10), Meter)
  assert mean(r)       == Quantity(4.5, 0.1 / math.sqrt(10), Meter)
  assert mean((1,2,3)) == 2


def scatter(it, m=None):
  """
  Calculates the scattering of an iterable around m, which defaults to the
  non-weighed mean of the iterable. The square of scattering is sometimes
  called the empirical variance. The scattering s is calculated by

    s = sqrt( 1/(n-1) sum (x - m)^2 ).

  The scattering of values can be used to estimate the statistical error of
  values, or be compare with the calculated statistical error. 

  Parameters:
    it  - Iterable which contains the values. This can also be Quantity
          object. Its standard deviations are ignored.
    m   - This is the mean around which the scattering should be calculated,
          see equation above. The default value is None, which causes the
          method to calculate the non-weighted mean of the value.

  Returns:
    A single value with the scattering. When a Quantity object was given,
    the initial unit will be preserved. A standard deviation of the
    scattering is not calculated.
  """
  if isinstance(it, np.ndarray):
    if m is None: m = sum(it) / len(it)
    return math.sqrt( ((it-m)**2).sum() / (len(it) - 1))
  elif isinstance(it, Quantity):
    return Quantity(scatter(it.value, m), unit=it.uvec)
  elif isinstance(it, collections.Iterable):
    return scatter(np.array(it), m)
  else:
    raise TypeError('First argument must be iterable.')

if __name__ == '__main__':
  assert scatter(x) == math.sqrt(((x-4.5)**2).sum() / (len(x) - 1))
  assert scatter(x, 10) == math.sqrt(((x-10)**2).sum() / (len(x) - 1))
  assert scatter(q) == math.sqrt(((x-4.5)**2).sum() / (len(x) - 1))
  assert scatter((1,2,3)) == 1
    
"""
def fit(func, x, y, sx=None, sy=None, p0=None):
  # p1, etc... must be Quantity objects
  ps = (p1, p2, p2, ...)
  m = ModelFit(func, ps)
  m.fit(x, y, sx, sy)
  

def const(x, c): return c



  
class ModelFit(object):

  def __init__(self, *parameters, func=const):
    self.func = func
    self.parameters = parameters

  def covmatrix():
    return self.cov

  def func(x, *pvalues):
    return self.func(x, *pvalues)
  
  def fit(self, x, y, sx=None, sy=None, covx=None, covy=None, p0=None):
    " ""
    returns chi^2/ndf
    " ""
    pass


  def save(self, filename):
    pass


  def figure(self):
    fig=1
    #make fig
    return fig

  def style():
    pass

  def estimate(self, x, y):
    return np.zeros(len(self.parameters)) + 1


def peak(data):
  pass

class SinLFit(ModelSin):
  pass

class SinCFit(ModelSin):
  pass

class SinFit(ModelFit):

  def __init__(self, amplitude, frequency, phase):
    self.func = np.sin
    self.parameters = parameters

  def estimate(self, x, y):
    #make fft
    pass
    


"""
