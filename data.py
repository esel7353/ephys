#!/usr/bin/python3
################################################################################
#
# Copyright (C) 2013, Frank Sauerburger
#   published under MIT license (see below)
#
################################################################################
#
#  Real Experimental Data Representation
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
import re
import math
import scipy.misc as sm
import scipy.constants as sc
import collections

################################################################################
# Classes

class IncompatibleUnits(TypeError): pass

class ParticipantsAreNotIndependend(Exception): 
  def __str__(self):
    return """You instructed the Quantity class to perform a propagation of
    errors. This can only be done there when then the two errors are
    statistically independent. This exception is to remind you to be careful
    that all errors are independent. To silence this exception call
    'Quantity.yesIKnowTheDangersAndPromiseToBeCareful()' once every
    session."""

class Quantity(object):
  """
  This class is to represent real experimental data. This means a value,
  its standard deviation and a unit can be stored together. Due to excessive
  operator overloading it is easy to make physical calculations which also
  propagate the standard deviation (Gaussian Error Propagation) and
  propagate the unit.
  >>> x = Quantity(4, 0.1, Meter)
  >>> print(x)
  4 +- 0.1 m
  >>> print(x * 2)
  8 +- 0.2 m

  The units are represented as products of powers of the SI base
  units. Internally the exponents are stored as vectors. This guarantees the
  unit calculation.
  >>> (Watt / Volt).sunit()
  'A'

  Using numpy arrays this data type can be used to represent whole data
  sets. Other functions from the ephys package work together which this data
  type such as fit routines. Dealing with data sets requires some caution,
  because some operators do not work as when dealing with single-valued
  quantities.
  For example the single-value quantity x from above (neglecting the error)
  can alternatively be created with
  >>> x = 1 * Meter

  The multi-value quantity y using the values v must be created with the
  constructor.
  >>> v = numpy.arange(10)
  >>> y = Quantity(v, unit=Meter)

  That's because by typing 'v * Meter' the mul-method of the numpy array v is
  called which would make an numpy array of quantities. This is a waste of
  resources. It is much faster to store one numpy array inside one quantity
  object. Also many functions of the ephys package will not work with
  the numpy array of quantities structure.
  Alternatively one can prevent this by typing
  >>> y = Meter * y
  which is much less intuitive.

  Besides storing many values in a single quantity object, it is possible to
  store a numpy array of different standard deviations inside. When both
  (several values and standard deviations) are stored, their numpy arrays must
  be equal in length.

  NOTE: one quantity object can only store one unit.
  """

  # error dependence protection
  participantsAreIndepended = False  # see ParticipantsAreNotIndependend

  # base storage
  dim            = 0
  baseLabel      = []
  baseSymbol     = []
  baseUnprefixed = []
  baseLatex      = []
  baseScaled     = []
  basePersistant = [] # this is the inverse of unitless

  # prefix storage
  prefixLabel  = []
  prefixSymbol = []
  prefixFactor = []
  prefixLatex  = []

  # unit storage
  unitLabel  = []
  unitSymbol = []
  unitLatex  = []
  unitVec    = []
  unitFactor = []


  def baseDim(n):
    """
    IF YOU WANT TO USE A UNIT SYSTEM, WHICH CAN BE REPRESENTED IN SI UNITS,
    YOU DON'T NEED TO CARE ABOUT THIS METHOD.

    Set the number of base units. See doc string of addBase for more
    information. This must be set before any base, prefix or unit is added.
    If this is changes afterwards, the unit calculations will fail.
    
    USUALLY THIS IS ALREADY SET WHEN THE THIS MODULE IS LOADED.
    """
    Quantity.dim = n
  
  def addBase(label, symbol, latex='', isscaled=False, unitless=False):
    """
    IF YOU WANT TO USE A UNIT SYSTEM, WHICH CAN BE REPRESENTED IN SI UNITS,
    YOU DON'T NEED TO CARE ABOUT THIS METHOD.

    All units are represented as a product of powers of the base units. Lets
    assume we have the 4-dim base with Meter(m), Second(s), Kilogram(kg) and
    Kelvin(K). Any arbitrary unit can then be represented by a product like 
      
      any unit = m^a * s^b * kg^c * K*d

    Internally only the vector (a, b, c, d) is stored. The units m, s, kg
    and K are called base to make the analogy to a vector space more
    obvious. 
    In real application the base has a higher dimension n. Every
    representation of a units is just a vector in this n-dim vector space.
    The unit calculations do not need the base vectors. The labels and
    symbols are need when the units should be printed.

    To name a base vectors, you can call this method addBase. The dimension
    of the base must be set in advance.

    Parameters:
      label    - Label of the base unit. This is the name of the unit, a
                 word, and not just a letter. Example: Meter, Second, ...

      symbol   - This is the short form of the base unit, a one or few
                 letter abbreviation. The the base is scaled, the prefix must be
                 inclueded. Example: m, s, kg, ...

      latex    - Latex equivalent of the symbol. The latex representation
                 should be accurate in contrast to the (ascii) symbol
                 representation which fails for example when dealing which
                 Greek letters. Default: ''
                 
      isscaled - Set this to True of the base unit has a prefix. The
                 prefactor will be detected with the prefix list. For
                 example Kilogram has a prefix. Default: False

      unitless - Set this to True, if the base unit is equal to unity. For
                 example Radian is equal to unity. The terminology for this
                 is 'unitless'. Default: False

    Returns:
      A named Quantity object with value 1 and no standard deviation.

    USUALLY THE SI BASE IS ALREADY ADDED, WHEN THIS MODULE IS LOADED.
    """
    if len(Quantity.baseSymbol) >= Quantity.dim:
      raise RuntimeError('All base units already added!')
      
    if isscaled and len(symbol) <= 1: raise ValueError('Scaled base symbol can not have length 1.')

    Quantity.baseSymbol.append(symbol)
    Quantity.baseLatex.append(latex)
    Quantity.baseLabel.append(label)
    Quantity.baseScaled.append(isscaled)
    Quantity.basePersistant.append(not unitless)
    if isscaled: Quantity.baseUnprefixed.append(symbol[1:])
    else:        Quantity.baseUnprefixed.append(None)


    # created base vector: (0, 0, ... 1, ... 0)
    # prevent float unit exponents, +-127 should be enough
    uvec = np.zeros(Quantity.dim, dtype='int8')
    uvec[len(Quantity.baseSymbol)-1] = 1

    return Quantity(1, unit=uvec).name(symbol, label, latex)

  def addPrefix(label, symbol, factor, latex=''):
    """
    IF YOU WANT TO USE A UNIT SYSTEM, WHICH CAN BE REPRESENTED IN SI UNITS,
    YOU DON'T NEED TO CARE ABOUT THIS METHOD.

    This method adds a prefix. To automatically detect prefixes, all allowed
    prefixes have to be added in advance.  A prefix for example is the
    preceding letter 'm' in 'ms' (Milli seconds). The prefix can be
    interpreted as a simple.
    factor. Here m=0.001.

    NOTE: PREFIXES ARE CON ONLY HAVE ONE LETTER SYMBOLS.

    Parameters:
      label    - Label of the prefix. This is the name of the prefix, a
                 word, and not just a letter. Example: Milli, Kilo, ...

      symbol   - This is the short form of the prefix, a one letter
                 abbreviation. Example: m, k, ...

      factor   - Numerical multiplication factor of the prefix. For Milli
                 this is supposed to be 1/1000, and for Kilo is should be 1000.
      
      latex    - Latex equivalent of the symbol. The latex representation
                 should be accurate in contrast to the (ascii) symbol
                 representation which fails for example when dealing with
                 Greek letters. Default: ''

    Returns:
      A named Quantity object with the factor as its value, an empty unit
      vector and no standard deviation.

    USUALLY ALL SI PREFIXES ARE ADDED WHEN THE MODULE IS LOADED.
    """
    if len(symbol) != 1:
      raise ValueError('Prefix symbol must be a single letter.')
    Quantity.prefixLabel.append(label)    
    Quantity.prefixSymbol.append(symbol)    
    Quantity.prefixLatex.append(latex)    
    Quantity.prefixFactor.append(factor)

    return Quantity(factor).name(symbol, label, latex)

  def addUnit(label, symbol, unit, latex=''):
    """
    Adds a units symbol/label/latex to the unit detection.

    All added units can be prefixed.

    Parameters:
      label    - Label of the unit. This is the name of the unit, a word,
                 and not just a letter. Example: Meter, Second, ...

      symbol   - This is the short form of the unit, a one or few letter
                 abbreviation. Example: m, s, kg, ...

      unit     - Representation of the unit. This can be whatever the
                 constructor of Quantity accepts.

      latex    - Latex equivalent of the symbol. The latex representation
                 should be accurate in contrast to the (ascii) symbol
                 representation which fails for example when dealing which
                 Greek letters. Default: ''

    Returns:
      A named Quantity object with value 1 and no standard deviation.
    """

    Quantity.unitLabel.append(label)
    Quantity.unitSymbol.append(symbol)
    Quantity.unitLatex.append(latex)
    # init ensures correct interpretation of given unit
    unit = Quantity(unit=unit)
    Quantity.unitVec.append(unit.uvec)
    Quantity.unitFactor.append(unit.value)

    return Quantity(unit=unit).name(symbol, label, latex)

  def name(self, symbol=None, label=None, latex=None):
    """
    Add a symbol, label and/or latex to a Quantity object. Only the given
    string will be overwritten, this means name() makes nothing.
    The object, will be changed in place:
    >>> U.name('Voltage', 'U')

    Parameters:
      label    - Label of the Quantity object. This is the name of the
                 quantity, a word, and not just a letter. Example: Length,
                 Voltage, ...
                 This can be a numpy array.
                 Default: None

      symbol   - This is the short form of the quantity, a one or few letter
                 abbreviation. Example: U, x1, x2
                 This can be a numpy array.
                 Default: None

      latex    - Latex equivalent of the symbol. The latex representation
                 should be accurate in contrast to the (ascii) symbol
                 representation which fails for example when dealing with
                 Greek letters and indices. Example: 'l_2', r'\omega'
                 This can be a numpy array.
                 Default: None

    Returns:
      The same object again (self). This way a cascading chain can be build
      up:
      >>> U = Quantity(1, Meter).name('l', 'length')
    """
    if symbol is not None:    self.symbol = symbol
    if label is not None:     self.label = label
    if latex is not None:     self.latex = latex

    return self


  def unitless(self):
    """
    Returns True if the unit vector is zero, or if all non vanishing
    components' bases  are marked as 'unitless'. Otherwise it returns False.

    For example Meter is not unitless, but Radian is unitless. The
    unitlessness of Radian is required because functions such as sin require
    a unitless argument, which is usually a multiple of Radian.
    """
    return not (self.uvec * Quantity.basePersistant).any() 

  def __init__(self, value=1, stddev=0, unit=1, variance=0,
               symbol='', label='', latex=''):
    """
    Creates a new quantity object. There are many different ways to call
    this method. First of all, all parameters are optional. The most common
    calls are 
    >>> Quantity(42, 0.1, Meter)
    >>> Quantity(42, unit=Meter)
    >>> Quantity(42, 0.1)
    >>> Quantity(42)

    Parameters:
      value     - The value of the Quantity object. This can be a single
                  value or an iterable. If stddev is also an iterable, they
                  must be equal in length.
                  Default: 1
                  
                  EXCEPTION: This can also be a string! In this case the
                  value is 1, and the string is treaded as a unit.

      stddev    - This is the standard deviation of the quantity. This can
                  also be a iterable. If so the quantity is treated as if it
                  has many values, which are all the same by coincidence. If
                  value is also a iterable, they both must be equal in length.
                  Defautl: 0

      unit      - The unit of the quantity. A multi-valued quantity can have
                  only ONE unit! Although this can be an iterable, which is
                  then treated as the unit vector. This can also be an other
                  Quantity object. Then the values will be multiplied, the
                  errors will be propagated (Gaussian error propagation) and
                  the unit will be copied.
                  This can also be a string. Then the method will try to
                  understand the unit based on the added base units,
                  prefixes and units. Only * and ^ are allowed. Spaced are
                  interpreted as *. Example: m^2 kg^-1 MeV or 5+-1 MeV. Latex syntax not
                  allowed.  This can also be an integer or float, then the
                  value and error are multiplied by this factor and the unit
                  vector is zero.
                  Default: 1

      variance  - The variance is identical to the square of the standard
                  deviation. Internally the uncertainty is stored as the
                  variance. This means the stddev argument is just to
                  shorten the call. The standard deviation overrides this
                  argument.
                  Default: 0

      symbol    - see Quantity.name, Default: ''
      label     - see Quantity.name, Default: ''
      latex     - see Quantity.name, Default: ''
  
  Return:
  The new Quantity object.
  """
    # Many operations here use Gaussian error propagation. It would be possible
    # to use the functionalities of the overloaded operators to do this, but
    # then there is the possibility of getting trapped in a loop, because the
    # overloaded operators also use this constructor. Therefore this function
    # should be self-sufficient.
    # Drawback: I have to write the same code again and again, what is error
    # prone.
    #
    # My opinon changed on 2014-01-10. The constructor can use the overloaded
    # operators, but not if the unit vector is given directly. Therefore the
    # overlaoded operators must only use the constructer form, where the unit
    # vector is passed.

    ########
    # name quantity
    self.symbol = symbol
    self.label  = label
    self.latex  = latex

    # EXCEPTION: if value is string, interpret as unit.
    if isinstance(value, str):
      unit = value
      value = 1
      
    ########
    # store value
    if isinstance(value, collections.Iterable) and not isinstance(value, np.ndarray):
      # iterable must be converted to numpy array
      self.value = np.array(value)
    else:
      # single value or already numpy array
      self.value  = value 


    ########
    # convert standard deviation to variance
    if isinstance(stddev, collections.Iterable) :
      # iterable must be converted to numpy array
      if not isinstance(stddev, np.ndarray): stddev = np.array(stddev)
      # if standard deviation is given, overwrite variance
      if (stddev > 0).any(): variance = stddev**2
    else:
      # if standard deviation is given, overwrite variance
      if stddev > 0: variance = stddev**2

    ########
    # store variance
    if isinstance(value, collections.Iterable) and not isinstance(value, np.ndarray):
      # iterable must be converted to numpy array
      self.variance = np.array(variance)
    self.variance = variance

    ########
    # interpret unit
    if isinstance(unit, Quantity):
      # >>> Quantity(unit=Meter)
      self.check(unit)  # check error independence
      self.uvec   = unit.uvec
      # gaussian error propagation
      if unit.variance > 0:
        self.variance = self.value**2 * unit.variance + unit.value**2 * self.variance
      else:
        self.variance = unit.value**2 * self.variance

      self.value *= unit.value

    elif isinstance(unit, str):
      # >>> Quantity(unit="m")
      tokens = re.split('[ *]', unit)
      self.uvec = np.zeros(Quantity.dim, dtype='int8')
      for t in tokens:
        t = t.strip()
        if len(t) == 0: continue

        if '^' in t: sym, exp = t.split('^')
        else:        sym, exp = t, '1'

        exp = float(exp)
          
        if exp % 1 == 0: exp = int(exp)
        # search hierarchy:
        #  (1) base symbol
        #  (2) unit symbol
        #  (3) prefix symbol + base symbol
        #  (4) prefix symbol + unit symbol
        #  (5) prefixed base
        #  (6) unprefixed base, e.g Gram
        #  (7) base label
        #  (8) unit label
        #  (9) prefix label + base label
        # (10) prefix label + unit label
        # (11) number
        if sym in Quantity.baseSymbol:
          # base symbol
          i = Quantity.baseSymbol.index(sym)
          self.uvec[i] += exp
        elif sym in Quantity.unitSymbol:
          # unit symbol
          i = Quantity.unitSymbol.index(sym)
          self.uvec     += Quantity.unitVec[i] * exp
          self.value    *= Quantity.unitFactor[i]**exp
          self.variance *= (Quantity.unitFactor[i]**exp)**2
        elif sym[0] in Quantity.prefixSymbol and len(sym) > 1 and sym[1:] in Quantity.baseSymbol:
          # prefix symbol + base symbol
          i = Quantity.baseSymbol.index(sym[1:])
          self.uvec[i] += exp
          j = Quantity.prefixSymbol.index(sym[0])
          self.value    *= Quantity.prefixFactor[j]**exp
          self.variance *= (Quantity.prefixFactor[j]**exp)**2
        elif sym[0] in Quantity.prefixSymbol and len(sym) > 1 and sym[1:] in Quantity.unitSymbol:
          # prefix symbol + unit symbol
          i = Quantity.unitSymbol.index(sym[1:])
          self.uvec     += Quantity.unitVec[i] * exp
          self.value    *= Quantity.unitFactor[i]**exp
          self.variance *= (Quantity.unitFactor[i]**exp)**2
          j = Quantity.prefixSymbol.index(sym[0])
          self.value    *= Quantity.prefixFactor[j]
          self.variance *= (Quantity.prefixFactor[j])**2
        elif len(sym) > 1 and sym[1:] in Quantity.baseUnprefixed:
          # prefixed base (e.g kg)
          i          = Quantity.baseUnprefixed.index(sym[1:])
          basePrefix = Quantity.baseSymbol[i][0]
          thisPrefix = sym[0]
          if basePrefix not in Quantity.prefixSymbol: 
            raise ValueError('Unit base {} has an unknown prefix.'.format(Quantity.baseSymbol[i]))
          basePrefix = Quantity.prefixSymbol.index(basePrefix)
          thisPrefix = Quantity.prefixSymbol.index(thisPrefix)
          basePrefix = Quantity.prefixFactor[basePrefix]
          thisPrefix = Quantity.prefixFactor[thisPrefix]

          self.uvec[i]  += exp
          self.value    *= (thisPrefix / basePrefix)**exp
          self.variance *= ((thisPrefix / basePrefix)**exp)**2
        elif sym in Quantity.baseUnprefixed:
          # (6) unprefixed base, e.g g for Gram
          i          = Quantity.baseUnprefixed.index(sym)
          basePrefix = Quantity.baseSymbol[i][0]
          if basePrefix not in Quantity.prefixSymbol: 
            raise ValueError('Unit base {} has an unknown prefix.'.format(Quantity.baseSymbol[i]))
          basePrefix = Quantity.prefixSymbol.index(basePrefix)
          basePrefix = Quantity.prefixFactor[basePrefix]

          self.uvec[i]  += exp
          self.value    *=  1 / basePrefix**exp
          self.variance *= (1 / basePrefix**exp)**2
        elif sym in Quantity.baseLabel:
          # base Label
          i = Quantity.baseLabel.index(sym)
          self.uvec[i] += exp
        elif sym in Quantity.unitLabel:
          # unit Label
          i = Quantity.unitLabel.index(sym)
          self.uvec     += Quantity.unitVec[i] * exp
          self.value    *= Quantity.unitFactor[i]**exp
          self.variance *= (Quantity.unitFactor[i]**exp)**2
        else:
          # number?
          try:
            sym = float(sym)
            self.value    *= float(sym)**exp
            self.variance *= (float(sym)**exp)**2
          except ValueError:
            if '+-' in sym:
              value, stddev = sym.split('+-')
              value = float(value)
              stddev = float(stddev)
              self.variance = self.value**2 * stddev**2 + value**2 * self.variance
              self.value *= value
            else: raise ValueError('Unknown token in unit string: {}'.format(sym))
          
    elif isinstance(unit, collections.Iterable):
      # >>> Quantity(unit=(1,0,0,0,0,0,0,0,0))
      if len(unit) != Quantity.dim:
        raise Exception('Unit vector length missmatch: {} dim base, but {} given!'.format(Quantity.dim, len(unit)))
      self.uvec = unit

    elif isinstance(unit, (int, float)):
      ### >>> Quantity(unit=1)
      self.value    *= unit
      self.variance *= unit**2
      self.uvec     = np.zeros(Quantity.dim, dtype='int8')
    else:
      raise NotImplemented('Unit parameter must eighter be a string, a quantity object, an iteratable or an int/float')


  def check(self, other):
    """
    Method internally used to check if the user has promised to be careful with correlated errors.
    If at least one of the two quantities, which the user wants to combine, has no standard deviation
    this check will silently do nothing, because one error can not be correlated.
    """
    var =  (self.variance * other.variance > 0)
    if (hasattr(var, 'any') and var.any()) or (not hasattr(var, 'any') and var):
      if not Quantity.participantsAreIndepended:
        raise ParticipantsAreNotIndependend()

  def yesIKnowTheDangersAndPromiseToBeCareful():
    """The Gaussian error propagation used here asumes that all the errors are
    independent. This might no be true, for example when you instruct python to
    do x * x. To remind you everytime you start to write a script, that the
    errors have to be independent, you have to call this function before
    doning any operation that uses Gaussian error propagtion of two Quantity
    objects. Otherwise an exception will be raised to remind you!
    """
    Quantity.participantsAreIndepended = True

  # Comparison Operators
  def __eq__(self, other):
    """
    Checks if values are equal/in agreement with in the standard deviation(s). 
    To check this the two values are subtracted (so Gaussian error propagation
    occurs if necessary) and checks if the resulting standard deviation is
    greater or equal than the resulting absolute value.
    """
    # >>> x == y
    # >>> x == 4.5
    x = self - other
    return x.variance >= x.value**2

  def __lt__(self, other): 
    """
    Checks if value is less and not in agreement with in the standard deviation(s). 
    To check this the two values are subtracted (so Gaussian error propagation
    occurs if necessary) and checks if the resulting standard deviation is
    less than the resulting absolute value.
    """
    # >>> x < y
    # >>> x < 4.5
    x = self - other
    return x.variance < x.value**2 and x.value < 0

  def __gt__(self, other): 
    """
    Checks if value is greater and not in agreement with in the standard deviation(s). 
    To check this the two values are subtracted (so Gaussian error propagation
    occurs if necessary) and checks if the resulting standard deviation is
    less than the resulting absolute value.
    """
    # >>> x > y
    # >>> x > 4.5
    x = self - other
    return x.variance < x.value**2 and x.value > 0

  def __ne__(self, other):   return not self == other
  def __ge__(self, other):   return self == other or self > other
  def __le__(self, other):   return self == other or self < other
                                   
  def __add__(self, other):
    """
    Add two quantity objects or a quantity object and a number. If standard
    deviations are given they will be propagated. If two quantity objects are
    to be summed, all non vanishing parts of the unit vector have to be equal,
    or an exception will be raised.
    """
    if isinstance(other, Quantity):
      # >>> x + y
      self.check(other)
      if ((self.uvec != other.uvec) * Quantity.basePersistant).any():  
        raise IncompatibleUnits('Can not sum {} and {}.'.format(self.sunit(), other.sunit()))
      value  = self.value + other.value
      # Gaussian error propagation  [2]
      variance = self.variance + other.variance
      return Quantity(value, variance=variance, unit=self.uvec)
    else:
      # >>> x + 5
      return self + Quantity(other, unit=self.uvec)
      
  def __sub__(self, other):
    """
    Subtract a number or a quantity object from an other quantity object. If standard
    deviations are given they will be propagated. If two quantity objects are
    to be subtracted, all non vanishing parts of the unit vector have to be equal,
    or an exception will be raised.
    """
    if isinstance(other, Quantity):
      # >>> x - y
      self.check(other)
      if ((self.uvec != other.uvec) * Quantity.basePersistant).any():  
        raise IncompatibleUnits('Can not subtract {} from {}.'.format(other.sunit(), self.sunit()))
      value  = self.value - other.value
      # Gaussian error propagation  [2]
      variance = self.variance + other.variance
      return Quantity(value, variance=variance, unit=self.uvec)
    else:
      # >>> x - 5
      return self - Quantity(other, unit=self.uvec)
      
  def __mul__(self, other):
    """
    Multiply two quantity objects or a quantity objects and a number. If
    standard deviations are given they will be propagated. If two quantity
    objects are to be multiplied, the unit vectors will added.  
    """
    if isinstance(other, Quantity):
      # >>> x * y
      self.check(other)
      value  = self.value * other.value
      # Gaussian error propagation  [2]
      variance = self.variance * other.value**2 +  other.variance * self.value**2
      uvec   = self.uvec + other.uvec
      return Quantity(value, variance=variance, unit=uvec)
    else:
      # >>> x * 43
      return self * Quantity(other)
      
  def __truediv__(self, other):
    """
    Divide a quantity object by a number or an other quantity object. If
    standard deviations are given they will be propagated. If two quantity
    objects are to be multiplied, the unit vectors will subtracted.  
    """
    if isinstance(other, Quantity):
      # >>> x / y
      self.check(other)
      value    = self.value / other.value
      variance = self.variance / other.value**2 + other.variance * self.value**2 / other.value**4
      uvec     = self.uvec - other.uvec
      return Quantity(value, variance=variance, unit=uvec)
    else:
      # >>> x / 7
      return self / Quantity(other)

  def __pow__(self, other):
    """
    Raise one quantity to the power of a number or of an other quantity object.
    Standard deviations will be propagated if given. If the power is a quantity
    object, it has to be unitless and can only be a numpy array if the
    raise quantity itself is unitless. Otherwise different values of the
    multi-valued result would have multiple units, which is not allowed.
    """
    a=self.value
    if isinstance(other, Quantity):
      # >>> x**y
      self.check(other)
      b=other.value
      if not other.unitless(): 
        raise IncompatibleUnits('Can not calculate the "{}-th power"!'.format(other.sunit()))
      if isinstance(b, np.ndarray) and not self.unitless():
        raise IncompatibleUnits('Can only raise to the numpy-array-power, if the raised value is unitless.')
      value     = a**b
      variance  = b**2 * a**(2*b-2) * self.variance + a**(2*b) * np.log(a)**2 * other.variance
      if isinstance(b, np.ndarray):
        uvec = np.zeros(Quantity.dim, dtype='int8')
      else:
        uvec = self.uvec*b
      return Quantity(value, variance=variance, unit=uvec)
    else:
      # >>> x**3
      return self**Quantity(other)

  def __mod__(self, other):
    """
    Add a relative standard deviation. The errors will be added using gaussian error
    propagation. If the relative error is a quantity object itself, the
    standard deviation of it will be ignored and it has to be unitless. Example
    >>> Quantity(5, 1) % 0.1

    Names (symbol, latex, label) will remain intact.
    """
    if isinstance(other, Quantity):
      # >>> x % y
      if not other.unitless():
        raise IncompatibleUnits('Relative error must not have a unit.')
      std = self.value * other.value
      return self + Quantity(0, stddev=std, unit=self.uvec, label=self.label, symbol=self.symbol, latex=self.latex)
    elif isinstance(other, (int, float)):
      # >>> x % 0.4
      return self % Quantity(other)

  def __or__(self, other):
    """
    Add an absolute standard deviation. The errors will be added using gaussian error
    propagation. If the relative error is a quantity object itself, the
    standard deviation of it will be ignored and it all the non vanishing parts
    of the unit vector must be equal. Example
    >>> Quantity(5, 1) | 2

    Names (symbol, latex, label) will remain intact.
    """
    if isinstance(other, Quantity):
      # >>> x | y
      try:
        return self + Quantity(0, stddev=other.value, unit=other.uvec, label=self.label, symbol=self.symbol, latex=self.latex)
      except IncompatibleUnits:
        raise IncompatibleUnits('Can not add absolute error given in {} for a quantity measured in {}.'.format(other.sunit(), self.sunit()))

    elif isinstance(other, (int, float)):
      # >>> x | 0.4
      return self | Quantity(other, unit=self.uvec)

  def removeError(self):
    """
    Removes all errors from the quantity object. The internal variance variable
    is set to zero. This changes the object in place and returns it again. The function was just
    added for completeness, I have no idea when one should ever be needing this.
    Names (symbol, latex, label) will remain intact.
    """
    self.variance = 0
    return self

  # Binary Right Operators

  def __radd__(self, other): 
    # >>> 4 + x
    return  self + other

  def __rsub__(self, other):
    # >>> 4 - x
    return -self + other

  def __rmul__(self, other):
    # >>> 4 * x
    return  self * other

  def __rtruediv__(self, other):
    # >>> 4 / x
    return Quantity(other) / self
    
  def __rpow__(self, other):        
    # >>> 4**x
    return Quantity(other)**self

  # Unitary Operators
  def __pos__(self): 
    # >>> +x
    return self

  def __neg__(self):
    # >>> -x
    return self * -1

  def __abs__(self):
    return Quantity(abs(self.value), variance=self.variance, unit=self.uvec)

  # Type Conversions
  def __complex__(self): return complex(self.value)
  def __int__(self):     return int(self.value)
  def __float__(self):   return float(self.value)

  def __str__(self): 
    """
    Returns a string representation  of the quantity.  Value and standard
    deviation is printed with full precision. For more physical output see
    Quantity.str and Quantity.latex.  If the quantity has a symbol (or a label,
    if no symbol is present) the name will precedent the values. The unit will
    added in the  Quantity.sunit format.
    If the quantity object is a multi-valued object, a table formatted string is returned.

    Examples:
      3.183098861837907 +- 0.6366197723675814 m^2 s^-2 kg
      U = 3.6787944117144233 +- 0.7357588823428847 m^2 s^-3 kg A^-1
      Current: 8.154845485377136 +- 1.103638323514327 A
    """
    if len(self) == 1:
      if self.symbol:
        naming = self.symbol + ' = '
      elif self.label:
        naming = self.label + ': '
      else:
        naming = ''
      return "{}{} +- {} {}".format(naming, self.value, self.stddev(), self.sunit())
    else:
      raise Exception('Multi-valued str() not yet implemented')

  def stddev(self):
    """
    Returns the standard deviation of the quantity object. This is a method,
    because internally the error is saved as the variance, which is the square
    of the standard deviation.
    """
    return np.sqrt(self.variance)

  def sunit(self):
    """
    Resurns a string prepresenation of its unit. The
    representations is a product of powers of the base
    units.
    Examples: 
      m^2 s^-2 kg       for Joule
      m^2 s^-3 kg A^-1  for Volt
      A                 for Ampere
    """ 
    rep = ""
    for b, p in zip(Quantity.baseSymbol, self.uvec):
      if p != 0:  # include only non-zero powers
        rep += b
        if p != 1: rep += "^" + str(p)  # include '^x' only if x is not one
        rep += " "
    return rep[:-1]  

  def __repr__(self): 
    """
    Creates a string which represents the object. If the string is evaled, it
    should create the same object (but at a different position in ram).
    Example:
    >>> Meter
    Quantity(1, 0.0, 'm', symbol='m', label='Meter')
    """
    naming = ''
    if self.symbol: naming += ", symbol=" + repr(self.symbol)
    if self.label: naming += ", label=" + repr(self.label)
    if self.latex: naming += ", latex=" + repr(self.latex)

    if self.uvec.any():
      return 'Quantity({}, {}, {}{})'.format(repr(self.value), repr(self.stddev()), repr(self.sunit()), naming)
    else:
      return 'Quantity({}, {}{})'.format(repr(self.value), repr(self.stddev()), naming)

  def __len__(self):
    """
    Checks if the data is still consistent i.e. all the properties are
    single-values or have one common lengths. This means that a property can be
    single-values or multi-valued. If it is multi-valued, it has to have the
    same lengths as any other multi-valued property. This includes the lengths of
    value, variance, symbol, label and latex. If they have different lengths, a
    ValueError is raised, otherwise the length is returned.
    """
    length = 1
    try:
      if hasattr(self.value, '__len__'):
        if length == 1: length = len(self.value)
        elif length != len(self.value): raise ValueError()
      if hasattr(self.variance, '__len__'):
        if length == 1: length = len(self.variance)
        elif length != len(self.variance): raise ValueError()
      # naming properties are always iterable because they are strings. To
      # checks if they are multi-valued, I test if they are not a string.
      if not isinstance(self.symbol, str) and hasattr(self.symbol, '__len__'):
        if length == 1: length = len(self.symbol)
        elif length != len(self.symbol): raise ValueError()
      if not isinstance(self.label, str) and hasattr(self.label, '__len__'):
        if length == 1: length = len(self.label)
        elif length != len(self.label): raise ValueError()
      if not isinstance(self.label, str) and hasattr(self.latex, '__len__'):
        if length == 1: length = len(self.latex)
        elif length != len(self.latex): raise ValueError()
    except ValueError:
      raise ValueError('Inconsistent data. The value, variance, symbol, label and/or latex variable are multi-valued but do not have equal lengths.')
    return length


  def __getitem__(self, key):
    """
    Returns the selected item/slice. All single-valued properties will ignore
    the key setting. If the quantity object is single valued, the item 0 can be
    indexed. This allows loops over single-valued quantity objects.
    [Also one can assume it is a 1x1x1...x1 multi-dim. array, because
    x[0][0][0][0][0][0]...[0] works]

    Calling methods of a sliced/index quantity can not change the
    quantity itself.
    """
    if len(self) == 1:  # as a side effect checks for consistent lengths
      try:
        (1,)[key] # check if the index only indexes the first item
      except IndexError:
        raise IndexError('Only on item can be index in a single valued quantity object.')
    
    # extract the selected properties if multi-valued, or returns the
    # single-value
    if hasattr(self.value, '__getitem__'): value = self.value[key]
    else: value = self.value

    if hasattr(self.variance, '__getitem__'): variance = self.variance[key]
    else: variance = self.variance
  
    # naming properties are always iterable because they are strings. To
    # checks if they are multi-valued, I test if they are not a string.
    if not isinstance(self.symbol, str): symbol = self.symbol[key]
    else: symbol = self.symbol
  
    if not isinstance(self.label, str): label = self.label[key]
    else: label = self.label
  
    if not isinstance(self.latex, str): latex = self.latex[key]
    else: latex = self.latex
  
    # build item/slice
    return Quantity(value, variance=variance, unit=self.uvec, symbol=symbol, label=label, latex=latex)

  def __setitem__(self, key, other):
    """
    Sets an item/slice. The item/slice can be set to an quantity object or a
    number/list. If an quantity object is assigned, all non-vanishing parts of
    the unit vector must be equal. 

    If a properties is not multi-valued, it will be converted into one if
    necessary. If the item/slice is not named and the quantity object has a
    single-valued naming, it will remain singly named. This decision is name
    independently for symbol, label and latex. Example: Assume you have a
    quantity object with multi-valued value and/or variance but single-valued
    symbol, label and latex. Now you assign an other quantity with symbol but no
    latex or label to an index. The resulting quantity will have multi-valued
    symbols but single valued label and latex.
    Long story short: The setitem will only override the naming, if a new naming
    is given; or you can not set an unnamed item in a named multi-valued
    quantity.
     
    If the quantity object is single valued, the contents can be replaced by
    indexing the first item.
    """

    if isinstance(other, Quantity):
      if ((self.uvec != other.uvec) * Quantity.basePersistant).any():  
        raise IncompatibleUnits('Can not assign item/slice measured in {} to a quantity measured in {}.'.format(other.sunit(), self.sunit()))

      l = len(self)  # as a side effect checks for consistent lengths
      ## single valued case
      if l == 1:
        try:
          (1, )[key]
        except IndexError:
          raise IndexError('Only the first item can be index in a single valued quantity object.')
        self.value = other.value
        self.variance = other.variance
        if other.symbol != '': self.symbol = other.symbol
        if other.label  != '': self.label = other.label
        if other.latex  != '': self.latex = other.latex
        return 


      ## multi-valued case
      # set value and variance
      if hasattr(self.value, '__setitem__') or self.value != other.value:
        if not hasattr(self.value, '__setitem__'): self.value = np.zeros(l) + self.value
        self.value[key] = other.value
      if hasattr(self.variance, '__setitem__') or self.variance != other.variance:
        if not hasattr(self.variance, '__setitem__'): self.variance = np.zeros(l) + self.variance
        self.variance[key] = other.variance

      # set symbol if a symbol is given
      # naming properties are always iterable because they are strings. To
      # checks if they are single-valued, I test if they are a string.
      if other.symbol != '':
        if isinstance(self.symbol, str): self.symbol = [self.symbol]*l 
        self.symbol[key] = other.symbol
        
      # set label if a label is given
      if other.label != '': 
        if isinstance(self.label, str): self.label = [self.label]*l
        self.label[key] = other.label

      # set latex if a latex is given
      if other.latex != '': 
        if isinstance(self.latex, str): self.latex = [self.latex]*l
        self.latex[key] = other.latex
    else:
      self[key] = Quantity(other, unit=self.uvec)
    
  def __delitem__(self, key):
    """
    Delete a slice  or an item. This will only affect multi-valued properties.
    An exception will be raised if the quantity is single-valued.
    """
    if len(self) == 1:  # as a side effect checks for consistent lengths
      raise ValueError('Can not delete slice or item from single valued quantity.') 
    if hasattr(self.value, '__delitem__'): del self.value[key]
    if hasattr(self.variance, '__delitem__'): del self.variance[key]

    # naming properties are always iterable because they are strings. To
    # checks if they are multi-valued, I test if they are not a string.
    if not isinstance(self.symbol, str): del self.symbol[key]
    if not isinstance(self.label, str): del self.label[key]
    if not isinstance(self.latex, str): del self.latex[key]

  def __iter__(self):
    """
    Return an iterator of the quantity. This also works for single valued
    quantity objects, so looping over every quantity is possible.
    """
    return Quantity.Iter(self)
    
  def calc(self, func, derivative=None, reqUnitless=True, propagateUnit=False, args=(), kwds={}, dx=0.001):
    """
    Calculates the result if the quantity is put into the given function. The
    errors will be propagated using Gaussian error propagation.

    Parameters:
      func            - The function where the quantity should be plugged in.
                        This must be callable. The args and kwds will be put
                        into the call.

      derivative      - The function which calculates the derivative of func.
       (None)           This must be callable and is used for Gaussion error
                        propagation. The function will also be called with args
                        and kwds. If this is omitted, the derivative will be
                        estimated using numerical methods, see dx parameter. 
                        
                        This can also be the numerical value of the derivative
                        evaluated at the value.

      reqUnitless     - Set this to false if the quantity doesn't has to be
       (True)           unitless.

      propagateUnit   - Determines whether the unit should be propagated in some
      (False=0)         way. If set to a int or float number, the unit vector
                        will be multiplied by the number. Otherwise this will be
                        called. Its only argument is the unit vector, its return
                        value is the new unit vector.

      args            - tuple of additional arguments which will be passed to
       ()               func and derivative.

      kwds            - dict of additional keywords which will be passed to func
       {}               and derivative.
       
      dx              - Need for numerical estimation of derivative. dx is
       0.001            passed to scipy.misc.derivative which does the
                        calculation.

    Returns:
      a new quantity with the result
    """

    if reqUnitless and not self.unitless():
      raise IncompatibleUnits('Argument of func must not have a unit.')
    if derivative is None:
      derivative = sm.derivative(func, self.value, dx=dx)
    elif not isinstance(derivative, (int, float)):
      derivative = derivative(self.value, *args, **kwds)
      
    value    = func(self.value, *args, **kwds)
    variance = derivative**2 * self.variance
      
    if isinstance(propagateUnit, (int, float)):
      uvec = propagateUnit * self.uvec
    else:
      uvec = propagateUnit(self.uvec)

    return Quantity(value, variance=variance, unit=uvec)



  def str(self): pass
  def latex(self): pass
  def store(self): pass
  def lexport(self): pass


  class Iter(object):
    """
    Iterator class for quantity.
    """
    def __init__(self, quantity):
      self.quantity = quantity
      self.i = -1 # pointing to item -1 (inced before referencing)
  
    def __iter__(self):
      """ python doc said an iterator must its own iterator """
      return self
  
    def __next__(self):
      """ return the next quantity item """
      self.i += 1
      if not self.i < len(self.quantity): raise StopIteration()
      return self.quantity[self.i]


################################################################################
def sin(x):
  if isinstance(x, Quantity): return self.calc(np.sin, np.cos, reqUnitless=True)
  else:                       return np.sin(x) 

def sinh(x):
  if isinstance(x, Quantity): return self.calc(np.sinh, np.cosh, reqUnitless=True)
  else:                       return np.sinh(x) 

def asin(x):
  def der(x): return 1 / np.sqrt(1-x**2)
  if isinstance(x, Quantity): return self.calc(np.asin, der, reqUnitless=True)
  else:                       return np.asin(x) 

def asinh(x):
  def der(x): return 1 / np.sqrt(1+x**2)
  if isinstance(x, Quantity): return self.calc(np.arcsinh, der, reqUnitless=True)
  else:                       return np.arcsinh(x) 


def cos(x):
  if isinstance(x, Quantity): return self.calc(np.cos, np.sin, reqUnitless=True)
  else:                       return np.cos(x) 

def cosh(x):
  if isinstance(x, Quantity): return self.calc(np.cosh, np.sinh, reqUnitless=True)
  else:                       return np.cosh(x) 

def acos(x):
  def der(x): return -1 / np.sqrt(1-x**2)
  if isinstance(x, Quantity): return self.calc(np.arccos, der, reqUnitless=True)
  else:                       return np.arccos(x) 

def acosh(x):
  def der(x): return 1 / np.sqrt(x**2-1)
  if isinstance(x, Quantity): return self.calc(np.arccosh, der, reqUnitless=True)
  else:                       return np.arccosh(x) 

def tan(x):
  def der(x): return 1 / np.cos(x)
  if isinstance(x, Quantity): return self.calc(np.cos, der, reqUnitless=True)
  else:                       return np.tan(x) 

def atan2(x, y):
  raise NotImplemented('there is still work to do')
  

def tanh(x):
  def der(x): return 1 / np.cosh(x)
  if isinstance(x, Quantity): return self.calc(np.tanh, der, reqUnitless=True)
  else:                       return np.tanh(x) 

def atan(x):
  def der(x): return 1 / (x**2 + 1)
  if isinstance(x, Quantity): return self.calc(np.arctan, der, reqUnitless=True)
  else:                       return np.arctan(x) 

def atanh(x):
  def der(x): return 1 / (1 - x**2)
  if isinstance(x, Quantity): return self.calc(np.arctanh, der, reqUnitless=True)
  else:                       return np.arctanh(x) 

def sqrt(x):
  if isinstance(x, Quantity): return self.calc(np.sqrt, propagateUnit=0.5)
  else:                       return np.sqrt(x) 

def exp(x):
  if isinstance(x, Quantity): return self.calc(np.exp, np.exp, reqUnitless=True)
  else:                       return np.exp(x) 

def log(x, base=math.e):
  def der(x): return 1 / (x * log(base))
  if isinstance(x, Quantity): return self.calc(np.log, der, reqUnitless=True, kwds={'base': base})
  else:                       return np.log(x) 

def log2(x):  return log(x, 2)
def log10(x): return log(x, 10)


################################################################################
# Init SI units

Quantity.baseDim(8)
Meter     = Quantity.addBase('Meter', 'm')
Second    = Quantity.addBase('Second', 's')
Kilogram  = Quantity.addBase('Kilogram', 'kg', isscaled=True)
Ampere    = Quantity.addBase('Ampere', 'A')
Kelvin    = Quantity.addBase('Kelvin', 'K')
Mol       = Quantity.addBase('Mol', 'mol')
Candela   = Quantity.addBase('Candela', 'cd')
Radian    = Quantity.addBase('Radian', 'rad', unitless=True)

Yocto = Quantity.addPrefix('Yocto', 'y', 1e-24)
Zepto = Quantity.addPrefix('Zepto', 'z', 1e-21) 
Atto  = Quantity.addPrefix('Atto', 'a', 1e-18)
Femto = Quantity.addPrefix('Femto', 'f', 1e-15)
Pico  = Quantity.addPrefix('Pico', 'p', 1e-12)
Nano  = Quantity.addPrefix('Nano', 'n', 1e-09)
Micro = Quantity.addPrefix('Micro', 'u', 1e-06, latex='\mu')
Milli = Quantity.addPrefix('Milli', 'm', 1e-03)
Centi = Quantity.addPrefix('Centi', 'c', 1e-02)
Deci  = Quantity.addPrefix('Deci', 'd', 1e-01)
Kilo  = Quantity.addPrefix('Kilo', 'k', 1e03)
Mega  = Quantity.addPrefix('Mega', 'M', 1e06)
Giga  = Quantity.addPrefix('Giga', 'G', 1e09)
Tera  = Quantity.addPrefix('Tera', 'T', 1e12)
Peta  = Quantity.addPrefix('Peta', 'P', 1e15)
Exa   = Quantity.addPrefix('Exa', 'E', 1e18)
Zetta = Quantity.addPrefix('Zetta', 'Z', 1e21)
Yotta = Quantity.addPrefix('Yotta', 'Y', 1e24)

Steradian         = Quantity.addUnit('Steradian', 'sr', Radian**2)
Newton            = Quantity.addUnit('Newton', 'N', Kilogram * Meter / Second**2) 
Joule             = Quantity.addUnit('Joule', 'J', Newton * Meter) 
Watt              = Quantity.addUnit('Watt', 'W', Joule / Second)
Volt              = Quantity.addUnit('Volt', 'V', Watt / Ampere)
Hertz             = Quantity.addUnit('Hertz', 'Hz', 1 / Second)
Pascal            = Quantity.addUnit('Pascal', 'Pa', Newton / Meter**2)
Coulomb           = Quantity.addUnit('Coulomb', 'C', Ampere * Second)
Ohm               = Quantity.addUnit('Ohm', 'Ohm', Volt / Ampere, latex='\Omega') 
Siemens           = Quantity.addUnit('Siemens', 'S', 1 / Ohm)
Farad             = Quantity.addUnit('Farad', 'F', Coulomb / Volt)
Henry             = Quantity.addUnit('Henry', 'H', Volt * Second / Ampere)
Weber             = Quantity.addUnit('Weber', 'Wb', Volt * Second)
Tesla             = Quantity.addUnit('Tesla', 'T', Weber / Meter**2)
Lumen             = Quantity.addUnit('Lumen', 'lm', Candela * Steradian)
Lux               = Quantity.addUnit('Lux', 'lx', Lumen / Meter**2)
Becquerel         = Quantity.addUnit('Becquerel', 'Bq', Hertz)
Gray              = Quantity.addUnit('Gray', 'Gy', Joule / Kilogram)
Sievert           = Quantity.addUnit('Sievert', 'Sv', Gray)
Degree            = Quantity.addUnit('Degree', '°', math.pi / 180 * Radian)
ArcMin            = Quantity.addUnit('ArcMin', '\'', Degree / 60)
ArcSec            = Quantity.addUnit('ArcSec', '\"', ArcMin / 60)
Liter             = Quantity.addUnit('Liter', 'l', Meter**3 / 1000)
Minute            = Quantity.addUnit('Minute', 'min', 60 * Second)
Hour              = Quantity.addUnit('Hour', 'hr', 60 * Minute)
Day               = Quantity.addUnit('Day', 'd', 24 * Hour)
Year              = Quantity.addUnit('Year', 'a', 365 * Day)
LightYear         = Quantity.addUnit('LightYear', 'ly', sc.light_year * Meter)
AstronomicalUnit  = Quantity.addUnit('AstronomicalUnit', 'au', sc.au * Meter)
Parsec            = Quantity.addUnit('Parsec', 'pc', sc.parsec * Meter)
Angstrom          = Quantity.addUnit('Ångström', 'Å', 1e-10 * Meter)
Dioptre           = Quantity.addUnit('Dioptre', 'dpt', 1 / Meter)
AtomicUnit        = Quantity.addUnit('AtomicUnit', 'u', sc.u * Kilogram)
ElectronVolt      = Quantity.addUnit('ElectronVolt', 'eV', sc.e * Joule)
Barn              = Quantity.addUnit('Barn', 'barn', 1e-28 * Meter**2)
PlanckConstant    = Quantity.addUnit('PlanckConstant', 'h', sc.h * Joule * Second)
DiracConstant     = Quantity.addUnit('DiracConstant', 'hbar', sc.hbar * Joule * Second)
SpeedOfLight      = Quantity.addUnit('SpeedOfLight', 'c', sc.c * Meter / Second)
Barn              = Quantity.addUnit('Barn', 'b', 10e-14 * (Centi * Meter)**2)
Fermi             = Quantity.addUnit('Fermi', 'fermi', Femto * Meter)