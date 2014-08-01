#!/usr/bin/python3
################################################################################
# Copyright (C) 2013-2014, Frank Sauerburger
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
from scipy.special import wofz
import collections
import warnings
import pylab
import matplotlib.pyplot as plt
import shelve
import re
import sys
from ephys.data import Quantity, Meter
import ephys.data as data
import ephys.texport

class StdDev(object):
  def __init__(self, q):
    self.q = q
  
  @property
  def uvec(self):
    return self.q.uvec

  def sprefunit(self):
    return self.q.sprefunit()

  @property
  def value(self):
    return np.sqrt(self.q.variance)
  
  @value.setter
  def value(self, v):
    self.q.variance = v**2
  

def daq(f, *columns, skip=0, sep=r'\s*[,]\s*', default=0., convert=float, c2d=False):
  l = len(columns)
  if l == 0: return
  for c in columns:
    if not isinstance(c, (Quantity, StdDev)):
      raise TypeError("Column must be quantity")

  val = []
  if isinstance(f, Quantity):
    raise TypeError("Fist argument must be file, not quantity!")
  elif isinstance(f, str):
    f = open(f, 'r')

  for line in f.readlines():
    if skip:
      skip -= 1
      continue
    line = line.strip()
    if not line: continue
    row = re.split(sep, line, l)

    if c2d:
      row = [r.replace(',', '.') for r in row]
    row = [ (convert(r) if r else default) for r in row]

    if len(row) < l: row += [default] * (l - len(row))

    val.append(row)
  f.close()

  val = list(zip(*val))

  result = []
  for i in range(l):
    f = float(Quantity(unit=columns[i].sprefunit()) / Quantity(unit=columns[i].uvec) )
    columns[i].value = np.array(val[i]) * f
  

################################################################################
# General
def readq(filename, label='', symbol='', latex='', unit='',  skip=0, sep=r"\s*(,|\s)\s*", default=0., convert=float, c2d=False, stddev=[], stdlink={}, n=0):
  """
  Reads a file of data. The file is supposed to contain columnes which are separated by some string. A row is defined
  by a line. Each columns will be read into a numpy array or quantity object. The function returns a list with all the columns.
  """
  f = open(filename)

  if isinstance(label, str):   label = [s.strip() for s in label.split('|')]
  if isinstance(symbol, str):   symbol = [s.strip() for s in symbol.split('|')]
  if isinstance(latex, str):   latex = [s.strip() for s in latex.split('|')]
  if isinstance(unit, str):   unit = [s.strip() for s in unit.split('|')]
  l = max(len(label), len(symbol), len(latex), len(unit), len(stddev), n)
  if len(label) < l: label += [''] * (l - len(label))
  if len(symbol) < l: symbol += [''] * (l - len(symbol))
  if len(latex) < l: latex += [''] * (l - len(latex))
  if len(unit) < l: unit += [1] * (l - len(unit))
  if len(stddev) < l: stddev += [0] * (l - len(stddev))


  val = []

  for line in f.readlines():
    if skip:
      skip -= 1
      continue
    line = line.strip()
    row = re.split(sep, line, l)
    if not l: l = len(row)

    row = [ (convert(r) if r else default) for r in row]

    if len(row) < l: row += [default] * (l - len(row))

    val.append(row)

  columns = list(zip(*val))

  result = []
  for i in range(l):
    value = columns[i]
    u = unit[i]
    ll = label[i]
    lx = latex[i]
    s = symbol[i]
    std = stddev[i]
    if i in stdlink:
      q = Quantity(value, std, u, label=ll, symbol=s, latex=lx)
      k = stdlink[i]
      if unit[k]:
        q = q | Quantity(val[k], unit=unit[k])
      else:
        q = q | Quantity(val[k], unit=unit[i])
      result.append(q)
    else: 
      result.append(Quantity(value, std, u, label=ll, symbol=s, latex=lx))
  
  return result



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
    (m, stddev) = mean(it.value, it.stddev())
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

################################################################################
# Plotter

class Plot:

  def __init__(self):
    self._made      = False
    self._data     = []
    self._boxes    = []
    self._enlarge  = (0.02,0.02,0.02,0.02)
    self._grid     = True
    self._xscale   = 'linear'
    self._yscale   = 'linear'
    self._leg      = False
    self._xaxis    = None
    self._yaxis    = None
    self._fitcount = 0

  def xaxis(self, q): 
    self._made = False
    if not isinstance(q, Quantity):
      raise TypeError("xaxis argument must be Quantity!")
    self._xaxis = Quantity(q.sprefunit())
    self._xaxis.label = q.label
    self._xaxis.latex = q.latex
    self._xaxis.symbol = q.symbol
    return self
  def yaxis(self, q): 
    self._made = False
    if not isinstance(q, Quantity):
      raise TypeError("yaxis argument must be Quantity!")
    self._yaxis = Quantity(q.sprefunit())
    self._yaxis.label = q.label
    self._yaxis.latex = q.latex
    self._yaxis.symbol = q.symbol
    return self

  def makex(self):
    if self._xaxis is None: return ''
    lab = []
    if self._xaxis.label:
      lab.append(self._xaxis.label)
    sl = self._xaxis.latex or self._xaxis.symbol
    if sl: 
      lab.append("${}$".format(sl))
    punit = self._xaxis.sprefunit(latex=True)
    if punit:
      lab.append('in ${}$'.format(punit))
    return ' '.join(lab)
  def makey(self):
    if self._yaxis is None: return ''
    lab = []
    if self._yaxis.label:
      lab.append(self._yaxis.label)
    sl = self._yaxis.latex or self._yaxis.symbol
    if sl: 
      lab.append("${}$".format(sl))
    punit = self._yaxis.sprefunit(latex=True)
    if punit:
      lab.append('in ${}$'.format(punit))
    return ' '.join(lab)

  """  def xunit(self, s):
    self._made = False
    if self._xaxis is None:
      self._xaxis = Quantity()
    q = Quantity(s)
    self._xaxis.preferredUnit = q.preferredUnit
    self._xaxis.value = q.value
    self._xaxis.uvec = q.uvec
    return self
  def yunit(self, s):
    self._made = False
    if self._yaxis is None:
      self._yaxis = Quantity()
    q = Quantity(s)
    self._yaxis.preferredUnit = q.preferredUnit
    self._yaxis.value = q.value
    self._yaxis.uvec = q.uvec
    return self"""

  def xlabel(self, s, symbol=None, latex=None): 
    self._made = False
    if self._xaxis is None:
      self._xaxis = Quantity()
    self._xaxis.label = s
    if symbol is not None: self._xaxis.symbol = symbol
    if latex is not None:  self._yaxis.latex  = latex
    return self
  def ylabel(self, s): 
    self._made = False
    if self._yaxis is None:
      self._yaxis = Quantity()
    self._yaxis.label = s
    if symbol is not None: self._xaxis.symbol = symbol
    if latex is not None:  self._yaxis.latex  = latex
    return self

  def legend(self, on=True):
    self._made = False
    self._leg = on
    return self


  def data(self, x, y, fmt, errorbar, label=True, **kwds):
    self._made = False
    if label == True:
      kwds['label']=y.label
    elif label is not None:
      kwds['label']=label


    self._data.append( (x, y, fmt, errorbar, kwds) )
    
    if isinstance(x, Quantity):
      if self._xaxis is None:
        self.xaxis(x)
    
    if isinstance(y, Quantity):
      if self._yaxis is None:
        self.yaxis(y)
    return self

  def error(self, x, y, fmt='.k', markersize=3, label=True, ecolor='0.3', **kwds):
    self._made = False
    self.data(x, y, fmt=fmt, markersize=markersize, label=label, ecolor=ecolor, errorbar=True, **kwds)
    return self # cascade

  def points(self, x, y, fmt='.k', markersize=3, label=True, **kwds):
    self._made = False
    self.data(x, y, fmt=fmt, markersize=markersize, label=label, errorbar=False, **kwds)
    return self # cascade
    
  def line(self, x, y, fmt='-b', linewidth=1, label=True, **kwds):
    self._made = False
    self.data(x, y, fmt=fmt, linewidth=linewidth, label=label, errorbar=False, **kwds)
    return self # cascade

  def histoline(self, x, y, *args, **kwds):
    l = len(x)

    doubledX = np.zeros(2*l)
    doubledY = np.zeros(2*l)
    for i in range(0, l):
      doubledY[2*i]   = y.value[i]
      doubledY[2*i+1] = y.value[i]

    deltaX = x.value[1] - x.value[0]
    print(x.value)
    doubledX[0] = x.value[0] - deltaX / 2
    for i in range(0, l-1):
      doubledX[2*i+1]   = (x.value[i] + x.value[i+1]) / 2
      doubledX[2*i+2] = (x.value[i] + x.value[i+1]) / 2
    deltaX = x.value[-1] - x.value[-2]
    doubledX[-1] = x.value[-1] + deltaX / 2



    doubledX = [doubledX[0]] + list(doubledX) + [doubledX[-1]]
    doubledY = [0] + list(doubledY) + [0]

    nx = Quantity(doubledX, 0, x.uvec)
    nx.name(x.label, x.symbol, x.latex)
    nx.preferredUnit = x.preferredUnit
    ny = Quantity(doubledY, 0, y.uvec)
    ny.name(y.label,y.symbol,y.latex)
    ny.preferredUnit = y.preferredUnit
    
    return self.line(nx, ny, *args, **kwds)


  fitcolors = 'brgcmy'

  def fit(self, mf, box=True, bpos=None, xmin=None, xmax=None, chi2=True):
    if not isinstance(mf, ModelFit):
      raise TypeError("Argument of Plot.fit must be a ModelFit, but {} given.".format(type(mf)))

    self._fitcount += 1
    text = 'Fit: ${}$\n\n'.format(mf.eq())
    if mf.sr >= 3:
      text += 'fit failed ({})!\n'.format(mf.sr)
    for p in mf.parameters:
      text += '  ${}$\n'.format(p.tex())
    if chi2:
      text += '  $\chi^2/\mathrm{ndf} = ' +('{:.2f}$'.format(mf.chi))

    xmin = xmin or min(mf.xo.value)
    xmax = xmax or max(mf.xo.value)

    if box:
      self.box( (self._fitcount, text, mf.xo, mf.yo, bpos) )
    x = Quantity(np.linspace(xmin, xmax, 200), unit=mf.xo.uvec)
    p0 = [Quantity(p.value, 0, p.uvec) for p in mf.parameters]
    self.line(x, mf.func(x, *p0), fmt='-'+Plot.fitcolors[(self._fitcount-1)%6])
    return self


  def box(self, text):
    self._made = False
    self._boxes.append(text)
    return self

  def xlog(self, log=True):
    self._made = False
    self._xscale = 'log' if log else 'linear'
    return self
  
  def ylog(self, log=True):
    self._made = False
    self._yscale = 'log' if log else 'linear'
    return self

  def save(self, filename, **kwds):
    if not self._made: self.make()
    # dpi not set here anymore
    # do this in a matlibplotrc file, e.g.
    #  figure.dpi     : 300 
    plt.tight_layout()
    plt.savefig(filename, **kwds)
    return self
    
  def show(self):
    if not self._made: self.make()
    plt.show()
    return self

  def make(self, clf=True, fig=None, axes=None):
    if clf: plt.clf()
    if fig is None and axes is None: fig = plt.figure()
    if axes is None: axes = fig.add_subplot(111)

    xlim = None
    ylim = None

    for x, y, fmt, errorbar, d in self._data:
      if isinstance(x, Quantity):
        f = Quantity(unit=x.uvec) / self._xaxis
        if not f.unitless():
          warnings.warn('The ratio of x axis unit ({}) and real unit ({}) must be unit less.'.format(self._xaxis.siunit(), y.siunit()))
        f = float(f)
        sx = x.stddev() * f
        x  = x.value * f
      else:
        #f = float(1 / Quantity(self._xunit))
        #x  = x * f
        sx = 0

      if isinstance(y, Quantity):
        f = Quantity(unit=y.uvec) / self._yaxis
        if not f.unitless():
          warnings.warn('The ratio of preferred unit ({}) and real unit ({}) must be unit less.'.format(self._yaxis.siunit(), y.siunit()))
        f = float(f)
        sy = y.stddev() * f
        y  = y.value * f
      else:
        #f = float(1 / Quantity(self.ypreferredUnit))
        #y  = y * f
        sy = 0

      xmin = np.nanmin(x - sx)
      xmax = np.nanmax(x + sx)
      ymin = np.nanmin(y - sy)
      ymax = np.nanmax(y + sy)
      if xlim is None: xlim = [xmin, xmax]
      if ylim is None: ylim = [ymin, ymax]
      xlim[0] = min(xlim[0], xmin)
      xlim[1] = max(xlim[1], xmax)
      ylim[0] = min(ylim[0], ymin)
      ylim[1] = max(ylim[1], ymax)

      xerr = sx
      yerr = sy

      if errorbar:
        axes.errorbar(x, y, sy, sx, fmt=fmt, **d)
      else:
        axes.plot(x, y, fmt, **d)

    xdif = xlim[1] - xlim[0]
    ydif = ylim[1] - ylim[0]
    xlim[0] -= xdif * self._enlarge[0]
    xlim[1] += xdif * self._enlarge[1]
    ylim[0] -= ydif * self._enlarge[2]
    ylim[1] += ydif * self._enlarge[3]

    axes.set_xlim(*xlim)
    axes.set_ylim(*ylim)

    i = 0
    taken = [ [0]*2, [0]*2, [0]*2 ]
    for b in self._boxes:
      if isinstance(b, tuple):
        fitnum, b, x, y, bpos = b
        
      if isinstance(bpos, tuple): 
        x = bpos[0] / 3
        y = bpos[1] / 2
      else:
        penalty = []
        for x in range(3):
          for y in range(2):
            points = 0 
            total = 0
            for dx, dy, fmt, errorbar, d in self._data:
              fx = float(Quantity(unit=dx.uvec) / self._xaxis)
              fy = float(Quantity(unit=dy.uvec) / self._yaxis)
              idx1 = (dx.value * fx > (xlim[0] + xdif * (x/3 - 0.1)))
              idx2 = (dx.value * fx < (xlim[0] + xdif * (x+1.3) / 3))
              idx3 = (dy.value * fy > (ylim[0] + ydif * (y/2 - 0.1)))
              idx4 = (dy.value * fy < (ylim[0] + ydif * (y+1.2) / 2))
              idx = idx1 * idx2 * idx3 * idx4
              points += len(dx.value[idx]) / len(dx.value)
              total = len(dx.value)
            points += total * taken[x][y]
            penalty.append( (x, y, points) )
        x, y, points = min(penalty, key=lambda y: y[2])
        taken[x][y] += 1

      if y == 0:
        yalign = 'top'
      elif y == 1:
        yalign = 'bottom'
      y = ylim[0] + ydif * 0.5

      if x == 0:
        xalign = 'right'
      elif x == 1:
        xalign = 'center'
      elif x == 2:
        xalign = 'left'
      x = xlim[0] + 0.5 * xdif

      print(x, y)

      if self._fitcount > 1:
        b = '({}) {}'.format(fitnum, b)
        fx = Quantity(unit=x.uvec) / self._xaxis
        fy = Quantity(unit=y.uvec) / self._yaxis
        if not fx.unitless():
          warnings.warn('The ratio of preferred unit ({}) and real unit ({}) must be unit less.'.format(self._xaxis.siunit(), x.siunit()))
        if not fy.unitless():
          warnings.warn('The ratio of preferred unit ({}) and real unit ({}) must be unit less.'.format(self._yaxis.siunit(), y.siunit()))
        fx = float(fx)
        fy = float(fy)
        tx = x.value * fx
        ty = y.value * fy
        tx = (min(tx)+max(tx))/2
        ty = max(ty)
        ty += 0.05 * ydif

        axes.text(tx, ty, '({})'.format(fitnum), color=Plot.fitcolors[(fitnum-1)%6], horizontalalignment='center', verticalalignment='bottom')


      #plt.figtext(*xy, s=b,bbox=dict(facecolor='w', edgecolor='black', pad=10), multialignment='left', **align)
      axes.annotate(b, xy=(x, y), bbox=dict(facecolor='w', edgecolor='k', pad=10), multialignment='left', horizontalalignment=xalign, verticalalignment=yalign)


    if self._grid: pylab.grid()
    axes.set_xscale(self._xscale)
    axes.set_yscale(self._yscale)
    if self._leg: pylab.legend()

    if self.makex(): axes.set_xlabel(self.makex())
    if self.makey(): axes.set_ylabel(self.makey())

    self._made = True
    return self # cascade


def peak(data):
  pass

################################################################################
# Fit
class ModelFit(object):

  def push(self, id):
    ephys.texport.push('fit_{}_eq'.format(id), self.eq())
    ephys.texport.push('fit_{}_chi'.format(id), '{:.3f}'.format(self.chi))
    for i in range(len(self.parameters)):
      ephys.texport.push('fit_{}_{}'.format(id, i), self.parameters[i].tex())
  
  def __init__(self, *parameters, func=None, modeq=''):
    """
    #n, #x and #y can be used in modeq to denote the n-th parameter.
    """
    if func is not None: self.func = func
    for p in parameters:
      if not isinstance(p, Quantity):
        raise TypeError('Fit Parameters must be quantities')
    self.parameters = parameters
    self.model = odr.Model(lambda p, x: self.func(x, *p))
    self.modeq = modeq
    self.sr = -1
    self.chi = -1
    self.xo = None
    self.yo = None

  def covmatrix():
    return self.cov

  def func(x, *pvalues):
    return 0 

  """  def puStr(q):
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

    return pUnit"""

  def xtrim(self, x, y, r):
    if r is None: return x, y

    if isinstance(r, RPicker):
      r = r.getRange(x, y)
    l, u = r
    r = (l <= x.value) * (x.value <= u)
    return x[r], y[r]

  def fit(self, x, y, estimate=True, maxit=None, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    if not isinstance(x, Quantity) or not isinstance(y, Quantity):
      raise TypeError('Data must be quantities')

    if estimate: self.estimate(x, y)
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
      warnings.warn('ODR fit did fail! ' + str(stopreason))
      chi = -1
    else:
      for p, b, sb in zip(self.parameters, result.beta, result.sd_beta):
        p.value    = b
        p.variance = sb**2
      ndf = len(sx) - len(result.beta)
      if ndf == 0:
        chi = 0
      else:
        chi = result.sum_square/ndf

    
    self.cov = result.cov_beta

    # make a unit check

    self.chi = chi
    self.sr = stopreason
    self.info = result

    return self

  def eq(self, x=None, y=None):
    if x is not None:
      xsl = x.latex or x.symbol
    elif self.xo is not None:
      xsl = self.xo.latex or self.xo.symbol  or 'x'
    else:
      xsl = 'x'

    if y is not None:
      ysl = y.latex or y.symbol
    elif self.yo is not None:
      ysl = self.yo.latex or self.yo.symbol or 'y'
    else:
      ysl = 'y'

    s = self.modeq
    s = s.replace('#x', xsl)
    s = s.replace('#y', ysl)
    for i in range(len(self.parameters)):
      p = self.parameters[i]
      s = s.replace('#{}'.format(i), p.latex or p.symbol or r'\beta_{{{}}}'.format(i))
    return s

  def picker(self, x, y, id):
    if not isinstance(x, Quantity) or not isinstance(y, Quantity):
      raise TypeError('Data must be quantities')
    for i in len(self.parameters):
      self.parameters[i] = Picker('{}_{}'.format(id, i)).getValue(x, y)
    return self
      

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    if not isinstance(x, Quantity) or not isinstance(y, Quantity):
      raise TypeError('Data must be quantities')
    self.xo = x
    self.yo = y
    return self

  def __str__(self):
    text = '-'*80 + '\n'
    text += 'Fit: {}\n\n'.format(self.eq())
    if self.sr >= 3:
      text += 'fit failed ({})!\n'.format(self.sr)
    for p in self.parameters:
      text += '  {}\n'.format(p.str())
    text += '  chi^2/ndf = {:.2f}\n'.format(self.chi)
    text += '-'*80 + '\n'
    return text


class PolynomialFit(ModelFit):
  def __init__(self, *coef, y='#y = '):
    eq = []
    for i in range(len(coef)):
      if i == 0:
        eq.append('#{}'.format(i))
      elif i == 1:
        eq.append('#{} #x'.format(i))
      else:
        eq.append('#{0:} #x^{{ {0:} }}'.format(i))

    eq = y + (' + '.join(eq))
    
    super().__init__(*coef, modeq=eq)


  def func(self, x, *coef):
    r = 0
    for i in range(len(coef)):
      if i%2==0: sign = 1
      else:
        if isinstance(x, Quantity):
          sign = 1.*(x.value>0) -1.* (x.value<0)
        else:
          sign = 1.*(x>0) - 1.*(x<0)
      r += coef[i] * sign * abs(x)**i 
    return r

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)

    m = (y.value.max() - y.value.min()) / (x.value.max() - x.value.min())
    c = y.value.min() - m * x.value.min()
    #m = (y.value[-1] - y.value[0]) / (x.value[-1] - x.value[0])
    #c = y.value[0] - m * x.value[0]

    self.parameters[0].value = c
    self.parameters[1].value = m

    if len(self.parameters) > 2:
      for p in self.parameters[2:]:
        if p.value == 0: p.value = 0.1

    self.xo = x
    self.yo = y

    return self

PolynomFit = PolynomialFit

"""class GaussCFit(ModelFit):
  
  def __init__(self, A, mu, sigma, c, x='---', y=''):
    def func(x, A, mu, sigma, c):
      return A * np.exp(-0.5 * (x-mu)**2 / sigma**2) + c

    eq = r'{}\cdot \exp \left(-\frac{{1}}{{2}} ({}-{})^2 / {}^2\right) + {}'.format(A.latex or
    A.symbol, x, mu.latex or mu.symbol, sigma.latex or sigma.symbol,
    c.latex or c.symbol)
    
    super().__init__(A, mu, sigma, c, func=func, modeq=eq)

  def fit(self, x, y, estimate=True, maxit=None):
    self.modeq = self.modeq.replace('---', x.latex or x.symbol)
    return super().fit(x, y, estimate, maxit)

  def estimate(self, x, y):
    c   = min(y.value)
    A   = max(y.value) -min(y.value)
    mu  = max(zip(x.value, y.value), key=lambda a: a[1])[0]
    sigma = (max(x.value) - min(x.value)) * (sum(y.value-c)/len(y.value)/A) 
   
    self.parameters[0].value = A
    self.parameters[1].value = mu
    self.parameters[2].value = sigma
    self.parameters[3].value = c

    self.xo = x
    self.yo = y

    return self"""

class GaussFit(ModelFit):
  def func(self, x, A, mu, sigma):
    return A * data.exp(-0.5 * (x-mu)**2 / sigma**2)
  
  def __init__(self, A, mu, sigma, y='#y = '):
    eq = y + r'#0 \cdot \exp \left(-\frac{(#x-#1)^2}{2 #2^2} \right)'
    super().__init__(A, mu, sigma, func=self.func, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    A  = max(zip(y.value, abs(y.value)), key=lambda a: a[1])[0]
    mu  = max(zip(x.value, abs(y.value)), key=lambda a: a[1])[0]
    sigma = (max(x.value) - min(x.value)) * (sum(y.value)/len(y.value)/A) / 2
   
    self.parameters[0].value = A
    self.parameters[1].value = mu
    self.parameters[2].value = sigma

    self.xo = x
    self.yo = y

    return self

class GaussCFit(ModelFit):
  def func(self, x, A, mu, sigma, offset):
    return A * data.exp(-0.5 * (x-mu)**2 / sigma**2) + offset
  
  def __init__(self, A, mu, sigma, offset, y='#y = '):
    eq = y + r'#0 \cdot \exp \left(-\frac{(#x-#1)^2}{2 #2^2} \right) + #3'
    super().__init__(A, mu, sigma, offset, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    offset = (y.value[0] + y.value[-1]) / 2
    self.parameters[3].value = offset
    _y = Quantity()
    _y.value = y.value - offset
    mf = GaussFit(*self.parameters[:3])
    mf.estimate(x, _y)
    return self

class GaussCLFit(ModelFit):
  def func(self, x, A, mu, sigma, offset, b):
    return A * data.exp(-0.5 * (x-mu)**2 / sigma**2) + offset + b * x
  
  def __init__(self, A, mu, sigma, offset, b, y='#y = '):
    eq = y + r'#0 \cdot \exp \left(-\frac{(#x-#1)^2}{2 #2^2} \right) + #3 + #4 #x'
    super().__init__(A, mu, sigma, offset, b, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    self.xo = x
    self.yo = y
    dy = y.value[-1] - y.value[0]
    dx = x.value[-1] - x.value[0]
    self.parameters[4].value = dy/dx
    _y = Quantity()
    _y.value = y.value - dy/dx * (x.value-x.value[0])
    mf = GaussCFit(*self.parameters[:4])
    mf.estimate(x, _y)
    return self

class LorentzFit(ModelFit):
  def func(self, x, A, mu, gamma):
    return A * gamma**2 / 4 / ( (x-mu)**2 + gamma**2/4)
  
  def __init__(self, A, mu, gamma, y='#y = '):
    eq = y + r'#0 \frac{ #2^2 \!/ 4}{ (#x-#1)^2 \!+ #2^2\!/4}'
    super().__init__(A, mu, gamma, func=self.func, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    A  = max(zip(y.value, abs(y.value)), key=lambda a: a[1])[0]
    mu  = max(zip(x.value, abs(y.value)), key=lambda a: a[1])[0]
    sigma = (max(x.value) - min(x.value)) * (sum(y.value)/len(y.value)/A) / 2
   
    self.parameters[0].value = A
    self.parameters[1].value = mu
    self.parameters[2].value = sigma

    self.xo = x
    self.yo = y

    return self

class LorentzCFit(ModelFit):
  def func(self, x, A, mu, gamma, offset):
    return A * gamma**2 / 4 / ( (x-mu)**2 + gamma**2/4) + offset
  
  def __init__(self, A, mu, gamma, offset, y='#y = '):
    eq = y + r'#0 \frac{ #2^2 \!/ 4}{ (#x-#1)^2 \!+ #2^2\!/4} + #3'
    super().__init__(A, mu, gamma, offset, func=self.func, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    self.xo = x
    self.yo = y
    offset = (y.value[0] + y.value[-1]) / 2
    self.parameters[3].value = offset
    _y = Quantity()
    _y.value = y.value - offset
    mf = LorentzFit(*self.parameters[:3])
    mf.estimate(x, _y)
    return self


class LorentzCLFit(ModelFit):
  def func(self, x, A, mu, gamma, offset, b):
    return A * gamma**2 / 4 / ( (x-mu)**2 + gamma**2/4) + offset + b * x
  
  def __init__(self, A, mu, gamma, offset, b, y='#y = '):
    eq = y + r'#0 \frac{ #2^2 \!/ 4}{ (#x-#1)^2 \!+ #2^2\!/4} + #3 + #4 #x'
    super().__init__(A, mu, gamma, offset, b, func=self.func, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    self.xo = x
    self.yo = y
    dy = y.value[-1] - y.value[0]
    dx = x.value[-1] - x.value[0]
    self.parameters[4].value = dy/dx
    _y = Quantity()
    _y.value = y.value - dy/dx * (x.value-x.value[0])
    mf = LorentzCFit(*self.parameters[:4])
    mf.estimate(x, _y)
    return self



class ErfFit(ModelFit):
  def func(self, x, A, mu, sigma):
    return A * data.erf((x-mu)/sigma)
  
  def __init__(self, A, mu, sigma, y='#y = '):
    eq = y + r'#0 \cdot \mathrm{erf} \left(\frac{#x-#1}{#2}\right)'
    super().__init__(A, mu, sigma, func=self.func, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    A   = (max(y.value) -min(y.value)) / 2
    lower= y.value < (max(y.value)+min(y.value))/2
    mu  = max(zip(x.value[lower], y.value[lower]), key=lambda a: a[1])[0]
    sigma = (max(x.value) - min(x.value)) / 2
    if y.value[0] > y.value[-1]:
      sigma *= -1

    self.parameters[0].value = A
    self.parameters[1].value = mu
    self.parameters[2].value = sigma


    self.xo = x
    self.yo = y

    return self

class ErfCFit(ModelFit):
  def func(self, x, A, mu, sigma, offset):
    return A * data.erf((x-mu)/sigma) + offset
  
  def __init__(self, A, mu, sigma, offset, y='#y = '):
    eq = y + r'#0 \cdot \mathrm{erf} \left( \frac{#x-#1}{#2}\right) + #3'
    super().__init__(A, mu, sigma, offset, func=self.func, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    self.xo = x
    self.yo = y
    offset  = y.value.mean()
    self.parameters[3].value = offset
    _y = Quantity()
    _y.value = y.value - offset
    mf = ErfFit(*self.parameters[:3])
    mf.estimate(x, _y)
    return self


class LorentzCLFit(ModelFit):
  def func(self, x, A, mu, sigma, offset, b):
    return A * data.erf((x-mu)/sigma) + offset + b * x
  
  def __init__(self, A, mu, sigma, offset, b, y='#y = '):
    eq = y + r'#0 \cdot \mathrm{erf} \left(\frac{#x-#1}{#2}\right) + #3 + #4 #x'
    super().__init__(A, mu, sigma, offset, b, func=self.func, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    self.xo = x
    self.yo = y
    dy = y.value[-1] - y.value[0]
    dx = x.value[-1] - x.value[0]
    self.parameters[4].value = dy/dx
    _y = Quantity()
    _y.value = y.value - dy/dx * (x.value-x.value[0])
    mf = ErfCFit(*self.parameters[:4])
    mf.estimate(x, _y)
    return self

class VoigtFit(ModelFit):
  def func(self, x, A, mu, gamma, sigma):
    z = ( (x-mu) + gamma * 1j) / (sigma * math.sqrt(2))
    if isinstance(z, Quantity): z = z.value
    return A* wofz(z).real #/ (sigma * math.sqrt(2*math.pi))

  def __init__(self, A, mu, gamma, sigma, y='#y = '):
    eq = y + r'\mathrm{Voigt}'
    super().__init__(A, mu, sigma, gamma,  func=self.func, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    A  = max(zip(y.value, abs(y.value)), key=lambda a: a[1])[0]
    mu  = max(zip(x.value, abs(y.value)), key=lambda a: a[1])[0]
    sigma = (max(x.value) - min(x.value)) * (sum(y.value)/len(y.value)/A) / 2
   
    self.parameters[0].value = A*2
    self.parameters[1].value = mu
    self.parameters[2].value = sigma/2
    self.parameters[3].value = sigma/2

    self.xo = x
    self.yo = y

    return self

class VoigtCFit(ModelFit):
  def func(self, x, A, mu, gamma, sigma, offset):
    z = ( (x-mu) + gamma * 1j) / (sigma * math.sqrt(2))
    if isinstance(z, Quantity): z = z.value
    return A*wofz(z).real +offset#/ (sigma * math.sqrt(2*math.pi)) + offset
  
  def __init__(self, A, mu, gamma, sigma, offset, y='#y = '):
    eq = y + r'\mathrm{Voigt} + #4'
    super().__init__(A, mu, gamma, sigma, offset, func=self.func, modeq=eq)

  def estimate(self, x, y, xrange=None):
    x, y = self.xtrim(x, y, xrange)
    self.xo = x
    self.yo = y
    offset = (y.value[0] + y.value[-1]) / 2
    self.parameters[4].value = offset
    _y = Quantity()
    _y.value = y.value - offset
    mf = VoigtFit(*self.parameters[:4])
    mf.estimate(x, _y)
    return self


class Gauge(data.Storable):
  def __init__(self, y, x, n=1):
    self.p = []
    self.x = x
    self.y = y
    xU = Quantity(x.siunit())
    yU = Quantity(y.siunit())
    for i in range(n+1):
      q = yU / xU**i
      q.name(latex='a_{}'.format(i))
      self.p.append(q)
    
    self.f = PolynomFit(*self.p)
    self.f.fit(x, y)
    
  def plot(self):
    p = Plot()
    p.error(self.x, self.y)
    p.fit(self.f)
    return p

  def __call__(self, x):
    q = self.f.func(x, *self.p)
    q.name(label=self.y.label, latex=self.y.latex, symbol=self.y.symbol)
    return q

class RPicker(object):
  storage = None

  def __init__(self, id):
    self.id = str(id)
    if not RPicker.storage:
      #try:
      RPicker.storage = shelve.open('{}.picker'.format(sys.argv[0]) )
      #except Exception:
      #  warnings.warn('Can not store range picks!') 


  def getRange(self, x, y):
    new = '--RPicker-new' in sys.argv[1:]
    new = new or '--RPicker-new-{}'.format(self.id) in sys.argv[1:]
    if (not new) and RPicker.storage and self.id in RPicker.storage:
      print(' *** RANGE PICKER for "{}" values restored'.format(self.id))
      r = RPicker.storage[self.id]
      print('     from {:.5g} to {:.5g}'.format(r[0], r[1]))
      return r
    else:
      return self.promt(x, y)

  def promt(self, x, y):
    p = Plot()
    p.error(x, y, ecolor='0.3')
    p.make()
    pylab.ion()
    p.show()
    pylab.ioff()
    print(' *** RANGE PICKER for "{}":'.format(self.id))
    if RPicker.storage is not None and self.id in RPicker.storage:
      r = RPicker.storage[self.id]
      print('     previously from {:.5g} to {:.5g}'.format(r[0], r[1]))

    xunit = p._xaxis.sprefunit()
    lower = input('     lower limit ({}) = '.format(xunit))
    upper = input('     upper limit ({}) = '.format(xunit))
    print('')

    lower = float(lower)
    upper = float(upper)

    f = Quantity(xunit) / Quantity(unit=x.uvec)
    f = float(f)
    lower *= f
    upper *= f

    if RPicker.storage is not None:
      RPicker.storage[self.id] = (lower, upper)
      print('     stored...')

    return lower, upper

class Table(object):
  """
  only for tex
  """

  def __init__(self, *cols, units=None, align=None):
    self.cols = cols
    if align is None:
      self.align = 'l'*len(cols)
    else:
      self.align = align

    if units is not None:
      if len(cols) != len(units):
        raise ValueError('Number of cols {} differs from number of units {}.'.format(len(cols), len(units)))
      self.units = []
      for u in units:
        if isinstance(u, str):
          self.units.append(Quantity(u))
        else:
          self.units.append(None)
    else:
      self.units = [None]*len(cols)
    self.rows = []

  def append(self, *items):
    if len(items) != len(self.cols):
      raise ValueError('Number of cols {} differs from number of items {}.'.format(len(self.cols), len(items)))
    for i, u in zip(items, self.units):
      if isinstance(u, Quantity) and not isinstance(i, Quantity):
        raise ValueError('Item for column with a units must be a Quantity')

    self.rows.append(items)
  
  def __str__(self):
    tab = []
    tab.append("  \\begin{tabular}{"+self.align+"}")
    head = []
    for h, u in zip(self.cols, self.units):
      if u is not None:
        head.append("{} in ${}$".format(h, u.sprefunit(latex=True)))
      else:
        head.append(h)
    tab.append(' & '.join(head) + r'\\\toprule[1.5pt]')

    for r in self.rows:
      row = []
      for i,u in zip(r, self.units):
        if i is None:
          row.append('')
        elif isinstance(u, Quantity):
            row.append('${}$'.format((i / u).tex()))
        elif isinstance(i, Quantity):
          row.append('${}$'.format(i.tex()))
        else:
          row.append(i)
      tab.append(' & '.join(row) + r'\\')
    tab.append('\\bottomrule[1.5pt] \end{tabular}')
    
    return '\n'.join(tab)

  def push(self, id):
    ephys.texport.push('tab_{}'.format(id), str(self))
    
     
    
