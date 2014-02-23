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
import scipy.odr as odr
import math
import scipy.constants as sc
import collections
import warnings
import pylab
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

    
"""
def fit(func, x, y, sx=None, sy=None, p0=None):
  # p1, etc... must be Quantity objects
  ps = (p1, p2, p2, ...)
  m = ModelFit(func, ps)
  m.fit(x, y, sx, sy)
  

def const(x, c): return c



"""  
class ModelFit(object):

  def __init__(self, *parameters, func=None, modeq=''):
    if func is not None: self.func = func
    for p in parameters:
      if not isinstance(p, Quantity):
        raise TypeError('Fit Parameters must be quantities')
    self.parameters = parameters
    self.model = odr.Model(lambda p, x: self.func(x, *p))
    self.modeq = modeq

  def covmatrix():
    return self.cov

  def func(x, *pvalues):
    return 0 


  def puStr(q):
    token = q.preferredUnit
    over = []
    under = []
    for t in token:
      p = t.exponent
      b = t.prefix + t.symbol
      b = '\\mathrm{' + b + '}'
      if p > 0:
        over.append(b + ('^'+str(p) if p>1 else ''))
      if p < 0:
        under.append(b + ('^'+str(-p) if p<-1 else ''))

    if len(under)>0:
      pUnit = '\\frac{' + (' '.join(over)) + '}{' + (' '.join(under)) + '}'
    else:
      pUnit = ' '.join(over)
    pUnit = r'\,' + pUnit

    return pUnit

  def fit(self, x, y, estimate=True, maxit=None, fullinfo=False):
    if not isinstance(x, Quantity) or not isinstance(y, Quantity):
      raise TypeError('Data must be quantities')

    if estimate:
      p0 = self.estimate(x, y)
    else:
      p0 = [p.value for p in self.parameters]

    self.xo = x
    self.yo = y


    sx = x.stddev()
    x  = x.value
    sy = y.stddev()
    y  = y.value

    if hasattr(x, '__len__'):    l = len(x)
    elif hasattr(y, '__len__'):  l = len(y)
    elif hasattr(sx, '__len__'): l = len(sx)
    else:                  l = __len__(sy)

    if not hasattr(sx, '__len__'): sx = np.zeros(l) + sx
    if not hasattr(x, '__len__'):  x  = np.zeros(l) +  x
    if not hasattr(sy, '__len__'): sy = np.zeros(l) + sy
    if not hasattr(y, '__len__'):  y  = np.zeros(l) +  y

    self.x = x
    self.y = y
    self.sx = sx
    self.sy = sy

    data = odr.RealData(x, y, sx, sy)
    algo = odr.ODR(data, self.model, p0, maxit=maxit)
    algo.run()
    result = algo.output
 
    stopreason = result.info
    if stopreason > 3:  # on error
      stopreason = result.stopreason
      warnings.warn('ODR fit did fail! ' + stopreason)

    
    dof = len(sx) - len(result.beta)
    self.cov = result.cov_beta
    for p, b, sb in zip(self.parameters, result.beta, result.sd_beta):
      p.value    = b
      p.variance = sb

    # make a unit check

    chi = result.sum_square/dof
    self.chi = chi
    self.sr = stopreason
    if fullinfo: 
      return chi, stopreason, result
    else:
      return chi, stopreason


  def plot(self, title="", clear=True):
    if clear: pylab.clf()
    pylab.xlim(min(self.x), max(self.x)) # add margin
    pylab.ylim(min(self.y), max(self.y)) # add margin

    xlab = []
    if self.xo.label: xlab.append(self.xo.label)
    if self.xo.latex:
      xlab.append('$' + self.xo.latex + '$')
    elif self.xo.symbol:
      xlab.append('$' + self.xo.symbol + '$')

    #xlab.append( #TODO
    if len(xlab): pylab.xlabel(' '.join(xlab))

    ylab = []
    if self.yo.label: ylab.append(self.yo.label)
    if self.yo.latex:
      ylab.append('$' + self.yo.latex + '$')
    elif self.yo.symbol:
      ylab.append('$' + self.yo.symbol + '$')
    if len(ylab): pylab.ylabel(' '.join(ylab))

    #einheit

    pylab.errorbar(self.x, self.y, self.sy, self.sx, '.k', markersize=3)
    x = np.linspace(min(self.x), max(self.x), 250)
    y = self.func(x, *[p.value for p in self.parameters])
    pylab.plot(x, y, '-b')

    
    leg = ['Fit: $' + self.modeq + '$'] + ['$' + p.tex() + '$' for p in self.parameters]
    leg.append(r'$\chi^2/\mathrm{dof} = ' + ('{:.3f}'.format(self.chi)) + '$')

    pylab.text(min(x), max(y), '\n'.join(leg), horizontalalignment='left',
    verticalalignment='top')

    return self

  def estimate(self, x, y):
    return np.zeros(len(self.parameters)) + 1

  def show(self):
    pylab.show()

  def savefig(self, *args, **kwds):
    pylab.savefig(*arg, **kwds)

class Plot:

  def __init__(self, title=""):
    pass
    



def peak(data):
  pass


class SinFit(ModelFit):

  def __init__(self, amplitude, frequency, phase):
    self.func = np.sin
    self.parameters = parameters

  def estimate(self, x, y):
    #make fft
    pass
    


class SinFit_L(SinFit):
  pass

class SinFit_C(SinFit):
  pass
