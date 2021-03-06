#!/usr/bin/python3
################################################################################
#
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
import re
import math
import scipy.misc as sm
import scipy.constants as sc
import collections
import shelve
import copy
import scipy.special as ss
from ephys.texport import push

SELF_PREFERRED = '--this should always fail when parsed-- 00'
SCALE_EQUALLY = '--this should always fail when parsed-- 01'
PRE_SCALE_ONLY = '--this should always fail when parsed-- 02'
DO_NOT_SCALE = False
SCALE_HOOD = '--this should always fail when parsed-- 04'
AUTO = '--this should always fail when parsed-- 05'

def q(unit, label='', latex='', symbol=''):
  """
  Shortcut for generating Quantity objects.

  This function is supposed to help everyone who is annoyed generating
  Quantity objects. This function should speed up the process of typing new
  Quantity objects. For example the very short command
  >>> q('m')
  Quantity(1, 0.0, 'm')

  generates a new length object. Also named object with values can be
  generated with this function as the following example illustrates.
  >>> q('42+-1 m', 'The Answer', 'A')
  Quantity(1, 0.0, 'm', label='The Answer', latex='A')

  Enjoy and save numerous keystrokes :)

  Parameter:
    unit      A unit string which must be parsable by
              Quantity._parseUnitString. This can contain values and error as
              shown in the example above. Mandatory!

    label     The label of the newly generated object. Default: None
    latex     The latex representation of the generating object. Default: None
    symbol    The symbol return of the newly generated object. Default: None

  Return:
    Returns a new Quantity object.

  See also:
    Length(), Time(), Temperature(), Force(), Energy(), ...
  """
  return Quantity(unit=unit, latex=latex, label=label, symbol=symbol)

################################################################################
# Classes

class Storable(object):

  """
  Interface for object which should be storage in a ephys.db file.

  The class works as an interface, which defines the functions, to store and
  restore objects in a shelve archive. Every class extending this class inherits
  the functions, such as Quantity. Therefore Quantity objects can be easily
  stored in 'ephys.db' and restored as the following example illustrates.
  >>> distance = Quantity('7 m')  # creating new objects
  >>> distance.store('Distance')  # storing object with an identifier
  ...
  >>> restoredDistance = Quantity.restore('Distance')
  """

  storage = None  # holding the shelve handler
                  # (note: this is located in the class)

  # TODO:
  # write function which ensures Storage.storage existence to reduce code
  # duplicity

  def restore(id):
    """
    Restores an object from the shelve archive ephys.db file.


    Restores an object from the shelve archive ephys.db file. Note: this
    function is located in the class! All objects are stored in a dictionary, so
    an identifier (string) denotes which objects should be read. An example
    calls therefore is
    >>> Quantity.restore('Distance')
    Quantity(7, 0.0, 'm')
    
    Parameter:
      id    the identifier (string) with which the objected was stored

    Return:
      the restored object
    """
    # create shelve handle if non existent
    if not Storable.storage: 
      Storable.storage = shelve.open('ephys.db')
    return Storable.storage[id]

  def store(self, id): 
    """
    Saves an object in the local shelve archie ephys.db.

    Saves an object in the local shelve archie ephys.db. Note: this function is
    located in the object!  The objects are stored in the file ephys.db which is
    an shelve archive. So the data is stored in a dictionary, and the object to
    store must be saved with an identifier. An example calls therefore is
    >>> distance = Quantity('7 m')  # creating new objects
    >>> distance.store('Distance')  # storing object with an identifier

    Parameter:
      id    the identifier (string) with which the objected was stored
    """
    # create shelve handle if non existent
    if not Storable.storage:
      Storable.storage = shelve.open('ephys.db')
    Storable.storage[id] = self

  def dir():
    """
    Print a list of all stored objects.

    Print a list of all stored objects. This function prints the key-repr pairs
    of all stored objects. If the raw data is needed (not printed output), you
    can access Storable.storage directly.

    See also:
      there might a console command 'ephysdir' which does calls this function
      form the command line.
    """
    # create shelve handle if non existent
    if not Storable.storage: 
      Storable.storage = shelve.open('ephys.db')
    for key in Storable.storage:
      print('{:16}->  {}'.format(key, repr(Storable.storage[key])))
  

class IncompatibleUnits(TypeError): 
  """
  Error class raised when two different units are summed, subtracted, ... .

  Error class, which is raised when ephys (in most cases the Quantity class)
  was instructed to perform an mathematical operation which is not possible with
  the units of the two operands. A very simple example is the summation of two
  Quantity objects with different units:
  >>> Quantity('m') + Quantity('s')

  This class has no body (just a pass statement). Its functionality is
  inherited form TypeError.
  """
  pass

class ParticipantsAreNotIndependend(Exception): 

  """
  Error class to ensure the independence when propagating errors.

  This error class is raised when two non-zero errors are supposed to be
  propagated. This is only valid, if the errors are statistically independent.
  This exception should remind the user to ensure the independent! To silence
  this exception, run
  >>> Quantity.yesIKnowTheDangersAndPromiseToBeCareful()

  once every session.
  """
  
  def __str__(self):
    return """You instructed the Quantity class to perform a propagation of
    errors. This can only be done when the two errors are statistically
    independent.
    
    This exception is to remind you to be careful, trhat all errors are
    independent.
    
    To silence this exception,  call:
    >>> Quantity.yesIKnowTheDangersAndPromiseToBeCareful()

    once every session.
    """

class Quantity(Storable):

  """
  Class representing real data with value, standard deviation and unit.

  This class is to represent real experimental data. This means a value,
  its standard deviation and a unit can be stored together. Due to excessive
  operator overloading it is easy to make physical calculations which also
  propagate the standard deviation (Gaussian Error Propagation) and
  propagate the unit.
  >>> x = Quantity(4, 0.1, Meter)
  >>> print(x)
  (4.00 +- 0.10) m
  >>> print(x * 2)
  (8.0 +- 0.2) m

  The units are represented as products of powers of the SI base
  units. Internally the exponents are stored as vectors. This guarantees the
  unit calculation.
  >>> (Watt / Volt).siunit()
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
  prefixDoNotUse  = []

  # unit storage
  unitLabel  = []
  unitSymbol = []
  unitLatex  = []
  unitVec    = []
  unitFactor = []

  storage = None


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

  def addPrefix(label, symbol, factor, latex='', doNotUse=False):
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
    Quantity.prefixDoNotUse.append(doNotUse)

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

    if True:
      if symbol in Quantity.unitSymbol:
        raise Exception('Can not add {} to unit base. 1'.format(symbol))
      if len(symbol) > 1 and symbol[0] in Quantity.prefixSymbol and symbol[1:] in Quantity.unitSymbol:
       raise Exception('Can not add {} to unit base. 2'.format(symbol))
      for p in Quantity.prefixSymbol:
        if p + symbol in Quantity.unitSymbol:
         raise Exception('Can not add {} to unit base. 3'.format(symbol))
          
    Quantity.unitLabel.append(label)
    Quantity.unitSymbol.append(symbol)
    Quantity.unitLatex.append(latex)
    # init ensures correct interpretation of given unit
    unit = Quantity(unit=unit)
    Quantity.unitVec.append(unit.uvec)
    Quantity.unitFactor.append(unit.value)

    return Quantity(unit=unit).name(symbol, label, latex)

  def name(self, label=None, latex=None, symbol=None):
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
    if symbol is not None:
      if isinstance(symbol, str): self.symbol = symbol
      else:                       self.symbol = list(symbol)

    if label is not None:
      if isinstance(label, str): self.label = label
      else:                      self.label = list(label)
     
    if latex is not None:
      if isinstance(latex, str): self.latex = latex
      else:                      self.latex = list(latex)

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

  def __init__(self, value=1, stddev=0, unit=1, variance=None,
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
                  This argument is overridden by the variance.
                  Defautl: 0

      unit      - The unit of the quantity. A multi-valued quantity can have
                  only ONE unit! Although this can be an iterable (except string
                  and quantity object), which is then treated as the unit
                  vector. This can also be an other Quantity object. Then the
                  values will be multiplied, the errors will be propagated
                  (Gaussian error propagation) and the unit will be copied.

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
                  shorten the call. The standard deviation is overridden by this
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
    self.preferredUnit = []
    if isinstance(symbol, str): self.symbol = symbol
    else:                       self.symbol = list(symbol)

    if isinstance(label, str): self.label = label
    else:                      self.label = list(label)

    if isinstance(latex, str): self.latex = latex
    else:                      self.latex = list(latex)

    # EXCEPTION: if value is string, interpret as unit.
    if isinstance(value, str):
      unit = value
      value = 1
      
    ########
    # store value
    if isinstance(value, collections.Iterable):
      # iterable must be converted to numpy array
      # the value will also be converted to a NEW numpy array if it also is a
      # numpy array, because the it is not referenced from outside. The array
      # might change, but references from outside presumably should not # change.
      self.value = np.array(value) * 1. # new numpy array is a float
    else:
      # single value or already numpy array
      self.value  = value 


    ########
    # error
    # if variance not given, use stddev.
    if variance is None:
      # convert standard deviation to variance and store
      if isinstance(stddev, collections.Iterable) :
        # iterable must be converted to numpy array, see value
        stddev = np.array(stddev) * 1.
        self.variance = stddev**2
      else:
        self.variance = stddev**2
    else:
      # store variance
      if isinstance(value, collections.Iterable):
        # iterable must be converted to numpy array, see value
        self.variance = np.array(variance) * 1.
      else:
        self.variance = variance

    ########
    # interpret unit
    if isinstance(unit, Quantity):
      # >>> Quantity(unit=Meter)
      self.check(unit)  # check error independence
      self.uvec   = unit.uvec
      self.variance = self.value**2 * unit.variance + unit.value**2 * self.variance

      self.value *= unit.value

    elif isinstance(unit, str):
      # >>> Quantity(unit="m")
      tokens = Quantity._parseUnitString(unit)
      self.uvec = np.zeros(Quantity.dim, dtype='int8')
      for t in tokens:
        if isinstance(t, _UnitToken):
          self.preferredUnit.append(t)
          self.variance *= t.prefactor**2
          self.value    *= t.prefactor
          self.uvec     += t.uvec
        elif isinstance(t, _VaerToken):
          self.variance = self.value**2 * t.stddev**2 + t.value**2 * self.variance
          self.value   *= t.value
          
    elif isinstance(unit, collections.Iterable):
      # >>> Quantity(unit=(1,0,0,0,0,0,0,0,0))
      if len(unit) != Quantity.dim:
        raise Exception('Unit vector length missmatch: {} dim base, but {} given!'.format(Quantity.dim, len(unit)))
      self.uvec = np.array(unit)
    elif isinstance(unit, (int, float)):
      ### >>> Quantity(unit=1)
      self.value    *= unit
      self.variance *= unit**2
      self.uvec      = np.zeros(Quantity.dim, dtype='int8')
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
    When one participant is a multi-valued, the result will be a numpy array with
    True or False, whether the values are equal or not.
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
    When one participant is a multi-valued, the result will be a numpy array with
    True or False, whether the values are less or not.
    """
    # >>> x < y
    # >>> x < 4.5
    x = self - other
    return (x.variance < x.value**2) * (x.value < 0)

  def __gt__(self, other): 
    """
    Checks if value is greater and not in agreement with in the standard deviation(s). 
    To check this the two values are subtracted (so Gaussian error propagation
    occurs if necessary) and checks if the resulting standard deviation is
    less than the resulting absolute value.
    When one participant is a multi-valued, the result will be a numpy array with
    True or False, whether the values are greater or not.
    """
    # >>> x > y
    # >>> x > 4.5
    x = self - other
    return (x.variance < x.value**2) * (x.value > 0)

  def __ne__(self, other):   return 1 - (self == other)
  def __ge__(self, other):   return (self == other) + (self > other)
  def __le__(self, other):   return (self == other) + (self < other)
                                   
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
        raise IncompatibleUnits('Can not sum {} and {}.'.format(self.siunit(), other.siunit()))
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
        raise IncompatibleUnits('Can not subtract {} from {}.'.format(other.siunit(), self.siunit()))
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
        raise IncompatibleUnits('Can not calculate the "{}-th power"!'.format(other.siunit()))
      if isinstance(b, np.ndarray) and not self.unitless():
        raise IncompatibleUnits('Can only raise to the numpy-array-power, if the raised value is unitless.')
      value     = a**b
      if (isinstance(other.variance, np.ndarray) and (other.variance==0).all()) or (not isinstance(other.variance, np.ndarray) and other.variance==0):
        variance  = b**2 * a**(2*b-2) * self.variance 
      else:
        variance  = b**2 * a**(2*b-2) * self.variance + a**(2*b) * np.log(np.abs(a))**2 * other.variance

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
      return (self + Quantity(0, stddev=std, unit=self.uvec)).name(label=self.label, symbol=self.symbol, latex=self.latex).prefunit(self.sprefunit())
    elif isinstance(other, (int, float, collections.Iterable)):
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
        return (self + Quantity(0, stddev=other.value, unit=other.uvec)).name(label=self.label, symbol=self.symbol, latex=self.latex).prefunit(self.sprefunit()) 
      except IncompatibleUnits:
        raise IncompatibleUnits('Can not add absolute error given in {} for a quantity measured in {}.'.format(other.siunit(), self.siunit()))

    elif isinstance(other, (int, float, collections.Iterable)):
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
  def __complex__(self):
    """ does not work for multi-valued quantities """
    return complex(self.value)

  def __float__(self):
    """ does not work for multi-valued quantities """
    return float(self.value)

  def __int__(self):
    """ does not work for multi-valued quantities """
    return int(self.value)

  def __str__(self): 
    """
    REDO
    Returns a string representation  of the quantity.  Value and standard
    deviation is printed with full precision. For more physical output see
    Quantity.str and Quantity.latex.  If the quantity has a symbol (or a label,
    if no symbol is present) the name will precedent the values. The unit will
    added in the  Quantity.sunit format.
    If the quantity object is a multi-valued object, no table formatted string is returned.

    Examples:
      3.183098861837907 +- 0.6366197723675814 m^2 s^-2 kg
      U = 3.6787944117144233 +- 0.7357588823428847 m^2 s^-3 kg A^-1
      Current: 8.154845485377136 +- 1.103638323514327 A
    """

    return self.str()
    """
    if len(self) == 1 or 1:
      if self.symbol:
        if isinstance(self.symbol, (list, tuple)):
          naming  = "["
          naming += ", ".join(self.symbol)
          naming += "] = "
        else:
          naming = self.symbol + ' = '
      elif self.label:
        if isinstance(self.label, (list, tuple)):
          naming  = "["
          naming += ", ".join(self.label)
          naming += "]: "
        else:
          naming = self.label + ': '
      else:
        naming = ''
      return "{}{} +- {} {}".format(naming, self.value, self.stddev(), self.siunit())
    else:
      raise Exception('Multi-valued str() not yet implemented')
    """

  def stddev(self):
    """
    Returns the standard deviation of the quantity object. This is a method,
    because internally the error is saved as the variance, which is the square
    of the standard deviation.
    """
    return np.sqrt(self.variance)

  def siunit(self, latex=False):
    """
    TODO latex
    Resurns a string prepresenation of its unit. The
    representations is a product of powers of the base
    units.
    Examples: 
      m^2 s^-2 kg       for Joule
      m^2 s^-3 kg A^-1  for Volt
      A                 for Ampere
    """ 
    over = []
    under = []
    for b, p in zip(Quantity.baseSymbol, self.uvec):
      if latex: b = r'\mathrm{' + b + '}'
      if p > 0:
        over.append(b + ('^'+str(p) if p>1 else ''))
      if p < 0:
        under.append(b + ('^'+str(-p) if p<-1 else ''))
    if len(over) == 0 and len(under) != 0: over.append('1')

    if latex:
      if len(under)>0:
        unit = '\\frac{' + (' '.join(over) or '1') + '}{' + (' '.join(under)) + '}'
      else:
        unit = ' '.join(over)
      unit = r'\,' + unit if unit else ''
    else:
      if len(under)>0:
        unit = (' '.join(over) or '1') + ' / ' + (' '.join(under))
      else:
        unit = ' '.join(over)

    return unit

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
      return 'Quantity({}, {}, {}{})'.format(repr(self.value), repr(self.stddev()), repr(self.siunit()), naming)
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
      if not isinstance(self.latex, str) and hasattr(self.latex, '__len__'):
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
    if hasattr(self.value, '__len__'): value = self.value[key]
    else: value = self.value

    if hasattr(self.variance, '__len__'): variance = self.variance[key]
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
    return Quantity(value, variance=variance, unit=self.uvec, symbol=symbol, label=label, latex=latex).prefunit(self.sprefunit()) 

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
        raise IncompatibleUnits('Can not assign item/slice measured in {} to a quantity measured in {}.'.format(other.siunit(), self.siunit()))

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

    # np.arrays do not support deleting...
    if hasattr(self.value, '__delitem__'):
       v = list(self.value) 
       del v[key]
       self.value = v

    if hasattr(self.variance, '__delitem__'):
       v = list(self.variance) 
       del v[key]
       self.variance = v

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

  def prefunit(self, s):
    """
    Sets the preferred unit of this quantity. The parameter must be parsable by
    Quantity._parseUnitString.
    The method returns self, so a cascade can but build.
    """
    unit = Quantity._parseUnitString(s)
    for u in unit:
      if isinstance(u, _VaerToken): raise ValueError('Preferred unit string must not contain value/error tokens.')

    self.preferredUnit = unit
    return self

  def _isaMagnitude(value, error):
    """
    All the internal methods dealing with a nice string representation
    should be prefixed with _isa. The prefix _isa is not an abbreviation. I
    just wanted to name this kid.
    """ 
    sv = math.floor(math.log10(abs(value))) if value else None
    se = math.floor(math.log10(abs(error))) if error else None
    # if first significant digit is a 1
    if se is not None:
      if error * 10**-se< 1.95: se-= 1

    return sv, se

  def _isaPenalty(value, error, ulist, svec):
    """
    This function should be minimized.
    """
    sv, se = Quantity._isaMagnitude(value, error)
    stilde = Quantity._isaScaleExponent(svec, ulist)

    if sv is not None: V = abs(sv + stilde - 2)**1.2
    else:              V = 0

    if se is not None: E = (se + stilde + 0)**2 / 4
    else:              E = 0 

    if sv is None: E *= 3
    if se is None: V *= 2


    U =  sum([abs(u.exponent**1.1 * s) for u, s in zip(ulist, svec[1:])])
    U += abs(svec[0])
    if abs(svec[0]) <= 2: U += 2
    #if sum([abs(s) for s in svec[1:]]) > 0:
    #U += sum([abs(s) for s in svec])/100

    return V + E + U

  def _isaScaleFactor(svec, ulist):
    return 10**(Quantity._isaScaleExponent(svec, ulist))

  def strunit(unitlist, latex=False):
    for t in unitlist:
      if isinstance(t, _VaerToken):
        raise ValueError("Unit specification must not contain a ValueToken.")
      assert(isinstance(t, _UnitToken))

    over = []
    under = []
    for t in unitlist:
      p = t.exponent
      if t.prefix == 'u' and latex:
        b = r'\mu ' + t.symbol
      else:
        b = t.prefix + t.symbol

      if latex:
        b = '\\mathrm{' + b + '}'
      if p > 0:
        if latex:
          over.append(b + ('^{'+str(p)+'}' if p>1 else ''))
        else:
          over.append(b + ('^'+str(p) if p>1 else ''))
      if p < 0:
        if latex:
          under.append(b + ('^{'+str(-p)+'}' if p<-1 else ''))
        else:
          under.append(b + ('^'+str(-p) if p<-1 else ''))

    if latex:
      if len(under)>0:
        pUnit = '\\frac{' + (' '.join(over) or '1') + '}{' + (' '.join(under)) + '}'
      else:
        pUnit = ' '.join(over)
    else:
      if len(under)>0:
        pUnit = (' '.join(over) or '1') + ' / ' + (' '.join(under))
      else:
        pUnit = ' '.join(over)

    return pUnit

  def sprefunit(self, latex=False):
    if self.preferredUnit:
      return Quantity.strunit(self.preferredUnit, latex)
    else:
      return self.siunit(latex)

  def _isaScaleExponent(svec, ulist):
    """
    General:
      (v+-e) * scaleFactor * 10^s_0 Π (10^s_i p_i u_i)^e_i
     
      v and e - are value and error
      s_0     - preunit scale exponent
      s_i     - unit scale exponents
      p_i     - the prefixes of the preferred unit
      u_i     - the preferred units
      e_i     - the exponents of the preferred unit
     
      10^s_i p_i will be combined to the new prefix
    """
    if len(svec) - 1 != len(ulist):
      raise ValueError('Number of unit scale exponents does not match the number of unit list items.')
    if len(svec) == 0:
      raise ValueError('There must be at least the PREunit scale exponent.')

    return -svec[0]-sum([s * u.exponent for s, u in zip(svec[1:],
    ulist)])

#  def _isaOptimatize(self, ....):

  def _isaNeighbors(svec, unitlist, hood):
    pfactors = []
    for f, nu in zip(Quantity.prefixFactor, Quantity.prefixDoNotUse):
      if not nu: pfactors.append(f)

    if 1 not in pfactors: pfactors.append(1)
    pexponents = sorted([math.log10(f) for f in pfactors])
    pexponentsNP = np.array(pexponents)
    plen = len(pexponents)
    neighbor = []
    L = len(svec)
    #if hood is None:
    #  hood = [prehood] + [unithood] * (L-1)
    l = sum(hood)
    hood = np.array(hood, dtype='int8')
    relHousenumber = -hood

    #pnumber = np.array([svec[0]] + [pexponents.index(s) for s in svec[1:]])

    upper = [svec[0]-1]
    lower = [svec[0]+1]
    for s, u in zip(svec[1:], unitlist):
      sprefix = 0
      if u.prefix:
        sprefix = math.log10(Quantity.prefixFactor[Quantity.prefixSymbol.index(u.prefix)])
      seff = s + sprefix
      greater = pexponentsNP[pexponentsNP > seff]
      less    = pexponentsNP[pexponentsNP < seff]
      if len(greater) > 0: upper.append(min(greater) - sprefix)
      else: upper.append(s)
      if len(less) > 0: lower.append(max(less) - sprefix)
      else: lower.append(s)
      

    while True:
      npn = upper * (relHousenumber==1) + svec * (relHousenumber==0) + lower * (relHousenumber==-1)
      neighbor.append(npn)

      # inc
      for i in range(len(relHousenumber)):
        if hood[i] == 1:
          relHousenumber[i] += 1
          if relHousenumber[i] >= 2:
            relHousenumber[i] = -1
          else:
            break

      if (relHousenumber == -hood).all(): # overflow occurred
        break

    return neighbor

  def _isaOptimize(value, error, svec, ulist, hood):
    threshold = 2
    prev = Quantity._isaPenalty(value, error, ulist, svec)
    init = prev
    mins = svec
    while True:
      neighbors = Quantity._isaNeighbors(mins, ulist, hood)
      environment = [ (n, Quantity._isaPenalty(value, error, ulist, n)) for n in neighbors]
      direction = min(environment, key=lambda e: e[1])
      if direction[1] >= prev:
        if prev + threshold < init:
          return direction[0]
        else:
          return svec
      else:
        mins, prev = direction

  def str(self, unit=SELF_PREFERRED, scale=SELF_PREFERRED,
  brackets=AUTO, name=True, reasonable=True, times=AUTO, hood=None,
  latex=False):
    """
    unit specification must be a string parsable by parserUnitString.
    if scale is False, value of prefix parameter will be ignored.
    """
    
    if len(self) > 1:
      l = []
      for q in self:
        l.append(q.str(unit, scale, brackets, name, reasonable, times, hood, latex))
      return '\n'.join(l)


    ######################################################################
    # make unit list
    if unit == SELF_PREFERRED: 
      token = copy.deepcopy(self.preferredUnit)
      # if this causes problems, if should only save a string
    else:
      token = Quantity._parseUnitString(unit)

    prefactor = 1
    uvec      = np.zeros(Quantity.dim, dtype='int8')
    for t in token:
      if isinstance(t, _VaerToken):
        raise ValueError("Unit specification must not contain a ValueToken.")
      assert isinstance(t, _UnitToken)
      prefactor *= t.prefactor
      uvec      += t.uvec
  
    if (self.uvec != uvec).any():
      extra = Quantity(unit=self.uvec-uvec).siunit()
      extra = Quantity._parseUnitString(extra)
      token.extend(extra)
  
    ######################################################################
    # make numbers

    value = self.value / prefactor
    error = self.stddev() / prefactor

    if reasonable:
      sv, se = Quantity._isaMagnitude(value, error)
      if se is not None:
        error = round(error, -se)
        value = round(value, -se)

    ssmult = None
    dig = None
    if scale:
      svec = [0] + [0]*len(token)
      if scale== SELF_PREFERRED:
        all = True
        for u in token:
          if u.sca != '!':
            all = False
            break
        if all:
          hood = [1] + [0]*len(token)
        else:
          hood = [0] + [u.sca != '!' for u in token]
        
      elif scale== SCALE_EQUALLY:
        hood = [0] + [1]*len(token)
      elif scale== PRE_SCALE_ONLY:
        hood = [1] + [0]*len(token)
      elif scale == SCALE_HOOD:
        pass
        # hood = hood...
      else: raise ValueError('Unknown scale mode!')

      svec = Quantity._isaOptimize(value, error, svec, token, hood) 
      value *= Quantity._isaScaleFactor(svec, token)
      error *= Quantity._isaScaleFactor(svec, token)
      if reasonable and se is not None:
          # error given
          dig = -se - Quantity._isaScaleExponent(svec, token)
          dig = int(dig)
          if dig < 0:
            value = round(value, -dig)
            error = round(error, -dig)
            dig = 0
           
      if svec[0] != 0:
        if latex:
          ssmult = '10^{'+str(int(svec[0]))+'}'
        else:
          ssmult = '10^'+str(int(svec[0]))

      ntoken = []
      for s, t in zip(svec[1:], token):
        eff = s
        if t.prefix:
          eff += math.log10(Quantity.prefixFactor[Quantity.prefixSymbol.index(t.prefix)])
        if eff != 0:
          sym = Quantity.prefixSymbol[Quantity.prefixFactor.index(10**eff)]
        else: sym = ''
        t.prefix = sym
        ntoken.append(t)
      token = ntoken

    if se is None and False:
      dig = -sv - Quantity._isaScaleExponent(svec, token)
      dig = int(dig)
      if dig < 0: 
        dig = 0
        value = round(value, -dig)
 

    ######################################################################
    # make unit str
    pUnit = Quantity.strunit(token, latex)
    if latex and pUnit:
      pUnit = '\\,' + pUnit
# TODO to be replaced by strunit()
    """
    over = []
    under = []
    for t in token:
      p = t.exponent
      b = t.prefix + t.symbol
      if latex:
        b = '\\mathrm{' + b + '}'
      if p > 0:
        over.append(b + ('^'+str(p) if p>1 else ''))
      if p < 0:
        under.append(b + ('^'+str(-p) if p<-1 else ''))

    if latex:
      if len(under)>0:
        pUnit = '\\frac{' + (' '.join(over)) + '}{' + (' '.join(under)) + '}'
      else:
        pUnit = ' '.join(over)
      pUnit = r'\,' + pUnit
    else:
      pUnit = ' '.join(over) + (' / ' + ' '.join(under) if len(under)>0 else '')
      """

    ######################################################################
    # make naming
    if name:
      if latex and self.latex: name = self.latex + ' ='
      elif self.symbol:        name = self.symbol + ' ='
      elif self.label and latex:
        name = '\\mathrm{' + self.label + '}:'
      elif self.label:
        name = self.label + ':'
      elif self.latex:        name = self.latex + ' ='

      else: name = ''
    else: name = ''

    ######################################################################
    # return
    if brackets == AUTO:
      if error == 0 or (not ssmult and not pUnit):
        brackets = False
      else: brackets = True

    if times == AUTO:
      if ssmult: times = '\\cdot' if latex else '*'
      else:      times = False

    if dig is not None: f = '{:.' + str(dig) + 'f}'
    else: f = '{}'

    l = []
    if name: l.append(name)

    if brackets:
      l.append('(' + f.format(value))
    else:
      l.append(f.format(value))

    if error != 0:
      if latex: l.append(r'\pm')
      else: l.append('+-')
      if brackets:
        l.append(f.format(error) + ')')
      else:
        l.append(f.format(error))

    if times: l.append(times)
    if ssmult: l.append(ssmult)
    if pUnit: l.append(pUnit)

    return ' '.join(l)


  def tex(self, unit=SELF_PREFERRED, scale=SELF_PREFERRED,
  brackets=AUTO, name=True, reasonable=True, times=AUTO, hood=None):

    return self.str(unit, scale, brackets, name, reasonable, times, hood,
    latex=True)


  
  def texport(self, id, *args, **kwds): 
    s = self.tex(*args, **kwds)
    s = r'\ensuremath{' + s + '}'
    push(id, s)


  def _parseUnitString(s):
    """
    This method is supposed to transform a given unit string into a list or a
    tuple which contains UnitToken and VaerToken objects which represent the
    individual tokens of the unit string.

    The unit string comply with the following rules. Otherwise an ValueError will
    be raised. The actual check and analysis is done using regular expressions,
    which might be slightly different. Refer to the source code for detailed
    information.
      - the syntax of a unit token is: [prefix]unit[^[-]number]
      - the syntax of a value/error token is: [-]number[+-number]
      - two tokens must be separated by either * or / or a white space
      - If the separator is a slash, ALL subsequent tokens' exponents will be
        multiplied by -1 until the next * separator is read.
        for example: m / s  J * kg is equal to m * kg / s / J
      - BRACKETS OF ANY SHAPE ARE NOT PERMITTED! (This feature might be added
        in later versions.)

    This method uses the method _searchUnit.

    Parameter:
      s - the string to be parsed

    Return:
      a list/tuple. The items will represent unit and vaer tokes using the
      corresponding classes.

    This method is only used internally!
    """
    l = []
    mode = 1

    # building regular expression
    # The regular expression will not check if the units and prefixed are known.
    rUnitToken = r'([!~]?)([a-zA-Z]+)(\s*\^\s*(-?[0-9]+(\.[0-9]+)?))?'
    rVaerToken = r'(-?[0-9]+(\.[0-9]*)?(e[-+]?[0-9]+)?)(\s*\+-\s*([0-9]+(\.[0-9]*)?(e[-+]?[0-9]+)?))?'
    rSeparator = r'\s*([\s*/])\s*'
    r          = '(({}|{})($|{}))*$'.format(rUnitToken, rVaerToken, rSeparator)
    # the re above might match expressions which end with a separator. I think the
    # one below suppresses this and allows white spaces in the beginning and at
    # the end.
    r          = '\s*(({}|{})(\s*$|{}(?!$)))*$'.format(rUnitToken, rVaerToken, rSeparator)

    # check overall format
    if not re.match(r, s): raise ValueError('Unit string is not properly formatted: {}'.format(s))

    # iterate over tokens
    for token in re.finditer('({}|{}|{})'.format(rUnitToken, rVaerToken, rSeparator), s):
      token = token.group(0)

      # analyse unit token
      m = re.match(rUnitToken, token)
      if m:
        sca = m.group(1)
        sym = m.group(2)
        exp = float(m.group(4) or 1)
        if exp % 1==0: exp = int(exp)
        # check if sym is valid unit
        prefix, symbol, prefactor, uvec = Quantity._searchUnit(sym)
        t = _UnitToken(prefix, symbol, mode*exp, prefactor**(mode*exp), uvec*exp*mode, sca)
        l.append(t)
        continue

      # analyse value error token
      m = re.match(rVaerToken, token)
      if m:
        val = float(m.group(1))
        err = float(m.group(5) or 0)
        if mode == -1:
          # if the value/error is in the denominator, the value and error has to
          # converted!
          # Let a be the value and sa the error. The resulting quantity f can be
          # calculated with f = c/a where c is the quantity before including
          # this factor. The calculation later will assume, that there was a
          # multiplication, say f = a' * c. This means we have to pretend that
          # here was a multiplication. It is easy to see, that a' = 1/a. What
          # about the error? The calculation later on will assume the formula:
          # sf = a' * c * sqrt(sa'^2/a'^2 + sc^2/c^2). To maintain this
          # relation, it is necessary that sa' = sa/a^2.
          err = err/val**2
          val = 1/val
        if val == 1 and err ==0: continue
        l.append(_VaerToken(val, err))
        continue

      # analyse separator
      m = re.match(rSeparator, token)
      if m:
        if m.group(1) == '/': mode = -1
        if m.group(1) == '*': mode = +1

    return l

  def _searchUnit(sym):
    """
    TODO 

    Search Hierarchy:
      (1) base symbol
      (2) unit symbol
      (3) prefix symbol + base symbol
      (4) prefix symbol + unit symbol
      (5) prefixed base
      (6) unprefixed base, e.g Gram
      (7) base label
      (8) unit label

      (9) prefix label + base label   -- not implemented!
     (10) prefix label + unit label   -- not implemented!

    Returns:
      uvec, prefactor, prefix, symbol

    """

    if len(sym) == 0: raise ValueError('Can not search empty symbol.')

    # base symbol
    if sym in Quantity.baseSymbol:
      i = Quantity.baseSymbol.index(sym)
      uvec = np.zeros(Quantity.dim, dtype='int8')
      uvec[i] = 1
      prefix = ''
      if len(sym) > 1 and sym[1:] in Quantity.baseUnprefixed:
        prefix = sym[0]
        sym = sym[1:]

      return prefix, sym, 1, uvec

    # unit symbol
    if sym in Quantity.unitSymbol:
      i = Quantity.unitSymbol.index(sym)
      return '', sym, Quantity.unitFactor[i], Quantity.unitVec[i]

    # prefix symbol + base symbol
    if sym[0] in Quantity.prefixSymbol and len(sym) > 1 and sym[1:] in Quantity.baseSymbol:
      i = Quantity.baseSymbol.index(sym[1:])
      j = Quantity.prefixSymbol.index(sym[0])
      uvec = np.zeros(Quantity.dim, dtype='int8')
      uvec[i] = 1
      return sym[0], sym[1:], Quantity.prefixFactor[j], uvec

    # prefix symbol + unit symbol
    if sym[0] in Quantity.prefixSymbol and len(sym) > 1 and sym[1:] in Quantity.unitSymbol:
      i = Quantity.unitSymbol.index(sym[1:])
      j = Quantity.prefixSymbol.index(sym[0])
      return sym[0], sym[1:], Quantity.prefixFactor[j] * Quantity.unitFactor[i], Quantity.unitVec[i]

    # prefixed base (e.g. Mg  for mega gram)
    if len(sym) > 1 and sym[1:] in Quantity.baseUnprefixed:
      i          = Quantity.baseUnprefixed.index(sym[1:])
      basePrefix = Quantity.baseSymbol[i][0]
      thisPrefix = sym[0]
      if basePrefix not in Quantity.prefixSymbol: 
        raise ValueError('Base unit {} has an unknown prefix.'.format(Quantity.baseSymbol[i]))
      basePrefix = Quantity.prefixSymbol.index(basePrefix)
      thisPrefix = Quantity.prefixSymbol.index(thisPrefix)
      basePrefix = Quantity.prefixFactor[basePrefix]
      thisPrefix = Quantity.prefixFactor[thisPrefix]

      uvec = np.zeros(Quantity.dim, dtype='int8')
      uvec[i] = 1
      return sym[0], sym[1:], thisPrefix / basePrefix, uvec

    # (6) unprefixed base (e.g. g for gram)
    if sym in Quantity.baseUnprefixed:
      i          = Quantity.baseUnprefixed.index(sym)
      basePrefix = Quantity.baseSymbol[i][0]
      if basePrefix not in Quantity.prefixSymbol: 
        raise ValueError('Base unit {} has an unknown prefix.'.format(Quantity.baseSymbol[i]))
      basePrefix = Quantity.prefixSymbol.index(basePrefix)
      basePrefix = Quantity.prefixFactor[basePrefix]

      uvec = np.zeros(Quantity.dim, dtype='int8')
      uvec[i] = 1
      return '', sym, 1 / basePrefix, uvec

    # base Label
    if sym in Quantity.baseLabel:
      i = Quantity.baseLabel.index(sym)
      uvec = np.zeros(Quantity.dim, dtype='int8')
      uvec[i] = 1
      sym = Quantity.baseSymbol[i]
      prefix = ''
      if len(sym) > 1 and sym[1:] in Quantity.baseUnprefixed:
        prefix = sym[0]
        sym = sym[1:]

      return prefix, sym, 1, uvec

    if sym in Quantity.unitLabel:
    # unit Label
      i = Quantity.unitLabel.index(sym)
      return '', Quantity.unitSymbol[i], Quantity.unitFactor[i], Quantity.unitVec[i]

    raise ValueError('Unit {} not found.'.format(sym))


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

def enumstr(s, *r):
  l = []
  for i in range(*r):
    l.append(s + str(i))
  return l

class _UnitToken(object):
  """
  This class is used for (code) efficient parsing and representation of unit
  strings. In fact this class one single term in a unit string. This class is
  intended to keep some data, and to not have some fancy methods!

  This class is only used internally!
  """
  def __init__(self, prefix, symbol, exponent, prefactor, uvec, sca):
    """
    Constructor of UnitToken class.
    
    Parameters:
      symbol    - The symbol as it appeared in the unit string but without any
                  prefix. This means that if the unit is prefixed, the prefix
                  is supposed to be removed an passed to prefix instead.

      uvec      - This is the unit vector representing the unit, i.e. a list
                  or tuple of Quantity.dim length.

      exponent  - The exponent as it appeared in the unit string, i.e. the
                  part after the '^' character. This should be an integer.
                  Default: 1

      prefix    - The string prefix is one was present. The concatenation of
                  symbol and prefix should always yield the string as it
                  appeared in the unit string. 

      prefactor - If the unit is prefixed, this must be its numerical
                  prefactor regarding the unit vector. The value should also
                  pay attention to the exponent! For example: when this
                  represents cm^2, the correct prefactor is 1/100^2

      sca       - Preference if scale behavior. '!' means, that the
                  isa algorithms does not try to scale this unit. '~'
                  means that the isa algorithms try to scale this unit
                  preferably. An empty string will mean that no preference
                  is set.
    """
    self.symbol = symbol
    self.uvec = uvec
    self.exponent = exponent
    self.prefix = prefix
    self.prefactor = prefactor
    self.sca = sca

  def __repr__(self):
    return '<Unit: {}{}`{}^{:+1.1f} | {:1.1f} * {} >'.format(self.sca or ' ', self.prefix or '_', self.symbol, float(self.exponent), self.prefactor, self.uvec)
                  
class _VaerToken(object):
  """
  This class is used for (code) efficient parsing and representation of unit
  strings. This class represents the 'a+-b' tokens in the string. This class
  is intended to keep only the valued a and b, and to not have some fancy
  methods!
  The name is an abbreviation of value and error.

  This class is only used internally!
  """
  def __init__(self, value, stddev):
    """
    Constructor of VaerToken class.

    Parameters: should be self explaining...
    """
    self.value = value
    self.stddev = stddev

  def __repr__(self):
    return '<Vaer: {} +- {} >'.format(self.value, self.stddev)

class BufferedQuantity(object):
  def __init__(self, unit="", symbol="", label="", latex=""):
    # test unit
    Quantity._parseUnitString(unit)
    self.unit = unit
    self.symbol = symbol
    self.label = label
    self.latex = latex
    self.value = []
    self.error = []

  def append(self, value, error=0):
    if isinstance(value, Quantity):
      ratio = Quantity(unit=value.uvec) / Quantity(self.unit)
      if ratio.unitless():  ratio = float(ratio)
      else: raise IncompatibleUnits("Can not append Quantity in {} to a Buffer in {}.".format(value.siunit(), self.unit))
      self.value.append(value.value * ratio)
      self.error.append(value.stddev() * ratio)
    else:
      self.value.append(value)
      self.error.append(error)
  
  def quantity(self):
    return Quantity(self.value, self.error, self.unit, label=self.label, latex=self.latex, symbol=self.symbol)


  def __len__(self):
    return len(self.value)

  def __getitem__(self, index):
    return Quantity(self.value[index], self.error[index], self.unit, label=self.label, latex=self.latex, symbol=self.symbol)

  def __repr__(self):
    return "<Buffer ({}): {} values measured in '{}' >".format(self.label or
    self.symbol or self.latex or 'Unnamed', len(self), self.unit)

  def __str__(self):
    q = self.quantity().str()
    return "Buffer:\n" + "\n".join( [" * " + line.strip() for line in q.split("\n")] )
    
################################################################################
def sin(x):
  if isinstance(x, Quantity): return x.calc(np.sin, np.cos)
  else:                       return np.sin(x) 

def sinh(x):
  if isinstance(x, Quantity): return x.calc(np.sinh, np.cosh)
  else:                       return np.sinh(x) 

def asin(x):
  der = lambda x: 1 / np.sqrt(1-x**2)
  if isinstance(x, Quantity): return x.calc(np.arcsin, der)
  else:                       return np.arcsin(x) 

def asinh(x):
  der = lambda x: 1 / np.sqrt(1+x**2)
  if isinstance(x, Quantity): return x.calc(np.arcsinh, der)
  else:                       return np.arcsinh(x) 

def cos(x):
  if isinstance(x, Quantity): return x.calc(np.cos, np.sin)
  else:                       return np.cos(x) 

def cosh(x):
  if isinstance(x, Quantity): return x.calc(np.cosh, np.sinh)
  else:                       return np.cosh(x) 

def acos(x):
  der = lambda x: -1 / np.sqrt(1-x**2)
  if isinstance(x, Quantity): return x.calc(np.arccos, der)
  else:                       return np.arccos(x) 

def acosh(x):
  der = lambda x: 1 / np.sqrt(x**2-1)
  if isinstance(x, Quantity): return x.calc(np.arccosh, der)
  else:                       return np.arccosh(x) 

def tan(x):
  der = lambda x: 1 / np.cos(x)**2
  if isinstance(x, Quantity): return x.calc(np.tan, der)
  else:                       return np.tan(x) 

def tanh(x):
  der = lambda x: 1 / np.cosh(x)**2
  if isinstance(x, Quantity): return x.calc(np.tanh, der)
  else:                       return np.tanh(x) 

def atan2(y, x):
  # atan and atan2 only differ by multiples of pi in the
  # 2pi modulus space: atan2 = +- atan + b pi
  # therefore abs(d/dx atan2) = abs(d/dx atan)
  if isinstance(x, Quantity) or isinstance(y, Quantity):
    r = y/x
    if not r.unitless():
      raise IncompatibleUnits('Argument of atan2 must have same unit.')
    val = np.arctan2(y.value, x.value)
    der = 1 / (r.value**2 + 1)
    std = der * r.stddev()
    return Quantity(val, std)
  else:
    return np.arctan2(y, x) 

def atan(x):
  der = lambda x: 1 / (x**2 + 1)
  if isinstance(x, Quantity): return x.calc(np.arctan, der)
  else:                       return np.arctan(x) 

def atanh(x):
  der = lambda x: 1 / (1 - x**2)
  if isinstance(x, Quantity): return x.calc(np.arctanh, der)
  else:                       return np.arctanh(x) 

def sqrt(x):
  der = lambda x: 1 / (2 * np.sqrt(x))
  if isinstance(x, Quantity): return x.calc(np.sqrt, der, propagateUnit=0.5, reqUnitless=False)
  else:                       return np.sqrt(x) 

def exp(x):
  if isinstance(x, Quantity): return x.calc(np.exp, np.exp)
  else:                       return np.exp(x) 

def erf(x):
  der = lambda x: 2 / math.sqrt(math.pi) * np.exp(-x**2)
  if isinstance(x, Quantity): return x.calc(ss.erf, der)
  else:                       return ss.erf(x) 

def log(x):
  der = lambda x: 1 / x
  if isinstance(x, Quantity): return x.calc(np.log, der)
  else:                       return np.log(x) 

def log2(x):
  der = lambda x: 1 / (x * log(2))
  if isinstance(x, Quantity): return x.calc(np.log2, der)
  else:                       return np.log2(x) 

def log10(x):
  der = lambda x: 1 / (x * log(10))
  if isinstance(x, Quantity): return x.calc(np.log10, der)
  else:                       return np.log10(x) 

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
Centi = Quantity.addPrefix('Centi', 'c', 1e-02, doNotUse=True)
Deci  = Quantity.addPrefix('Deci', 'd', 1e-01, doNotUse=True)
Kilo  = Quantity.addPrefix('Kilo', 'k', 1e03)
Mega  = Quantity.addPrefix('Mega', 'M', 1e06)
Giga  = Quantity.addPrefix('Giga', 'G', 1e09)
Tera  = Quantity.addPrefix('Tera', 'T', 1e12)
Peta  = Quantity.addPrefix('Peta', 'P', 1e15)
Exa   = Quantity.addPrefix('Exa', 'E', 1e18)
Zetta = Quantity.addPrefix('Zetta', 'Z', 1e21)
Yotta = Quantity.addPrefix('Yotta', 'Y', 1e24)

# to introduce a prefixed unit here makes no sense. I think is is not supported
# anyway.
Steradian         = Quantity.addUnit('Steradian', 'sr', Radian**2)
Newton            = Quantity.addUnit('Newton', 'N', Kilogram * Meter / Second**2) 
Joule             = Quantity.addUnit('Joule', 'J', Newton * Meter) 
Watt              = Quantity.addUnit('Watt', 'W', Joule / Second)
Volt              = Quantity.addUnit('Volt', 'V', Watt / Ampere)
Hertz             = Quantity.addUnit('Hertz', 'Hz', 1 / Second)
Pascal            = Quantity.addUnit('Pascal', 'Pa', Newton / Meter**2)
Bar               = Quantity.addUnit('Bar', 'bar', 100000 * Pascal)
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
Degree            = Quantity.addUnit('Degree', 'deg', math.pi / 180 * Radian, latex='{}^\circ')
ArcMin            = Quantity.addUnit('ArcMin', '\'', Degree / 60)
ArcSec            = Quantity.addUnit('ArcSec', '\"', ArcMin / 60)
Liter             = Quantity.addUnit('Liter', 'l', Meter**3 / 1000)
Minute            = Quantity.addUnit('Minute', 'min', 60 * Second)
Hour              = Quantity.addUnit('Hour', 'hr', 60 * Minute)
Day               = Quantity.addUnit('Day', 'd', 24 * Hour)
Year              = Quantity.addUnit('Year', 'yr', 365 * Day)
LightYear         = Quantity.addUnit('LightYear', 'ly', sc.light_year * Meter)
AstronomicalUnit  = Quantity.addUnit('AstronomicalUnit', 'AU', sc.au * Meter)
Parsec            = Quantity.addUnit('Parsec', 'psec', sc.parsec * Meter)
Angstrom          = Quantity.addUnit('Ångström', 'Å', 1e-10 * Meter)
Dioptre           = Quantity.addUnit('Dioptre', 'dpt', 1 / Meter)
AtomicUnit        = Quantity.addUnit('AtomicUnit', 'u', sc.u * Kilogram)
ElectronVolt      = Quantity.addUnit('ElectronVolt', 'eV', sc.e * Joule)
Barn              = Quantity.addUnit('Barn', 'barn', 1e-28 * Meter**2)
PlanckConstant    = Quantity.addUnit('PlanckConstant', 'h', sc.h * Joule * Second)
DiracConstant     = Quantity.addUnit('DiracConstant', 'hbar', sc.hbar * Joule * Second)
SpeedOfLight      = Quantity.addUnit('SpeedOfLight', 'c', sc.c * Meter / Second)
Fermi             = Quantity.addUnit('Fermi', 'fermi', Femto * Meter)


# Function with genrate a new quantity. This will save several
# keystrokes ;)
def Length(*nameing):        return Q('m', *nameing)
def Time(*nameing):          return Q('s', *nameing)
def Temperature(*nameing):   return Q('K', *nameing)
def Velocity(*nameing):      return Q('m/s', *nameing)
def Area(*nameing):          return Q('m^2', *nameing)
def Volume(*nameing):        return Q('m^3', *nameing)
def Pressure(*nameing):      return Q('N / m^2', *nameing)
def Density(*nameing):       return Q('kg / m^3', *nameing)
def Acceleration(*nameing):  return Q('m / s^2m', *nameing)
def Energy(*nameing):        return Q('J', *nameing)
def Voltage(*nameing):       return Q('V', *nameing)
def Current(*nameing):       return Q('A', *nameing)
def Power(*nameing):         return Q('W', *nameing)
def Resistance(*nameing):    return Q('Ohm', *nameing)
def Inductivity(*nameing):   return Q('Henry', *nameing)
def Charge(*nameing):        return Q('C', *nameing)
def Capacity(*nameing):      return Q('Farad', *nameing)
def Force(*nameing):         return Q('N', *nameing)
def Rate(*nameing):          return Q('Hz', *nameing)
