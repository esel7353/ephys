#!/usr/bin/python3

################################################################################
#
# Copyright (C) 2013, Frank Sauerburger
#   published under MIT license (see below)
#
################################################################################
# 
# Test suits of all functionalities of the ephys package.
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

from data import Quantity
import math
import data
import unittest
import numpy as np

class Misc(unittest.TestCase):
  def test_enumstr(self):
    self.assertEqual(data.enumstr('s_', 3), ['s_0', 's_1', 's_2']  )
    self.assertEqual(data.enumstr('s_', 4, 9, 2), ['s_4', 's_6', 's_8']  )

class QuantityScalerTest(unittest.TestCase):
  
  def setUp(self):
    self.METER = [1] + [0]*7
    Quantity.yesIKnowTheDangersAndPromiseToBeCareful()

  def assertQuantity(self, quantity, v, std, uvec=[0, 0, 0, 0, 0, 0, 0, 0], symbol='', label='', latex=''):
    self.assertAlmostEqual(quantity.value, v)
    self.assertAlmostEqual(quantity.variance, std**2)
    uvec = np.array(uvec)
    self.assertTrue((quantity.uvec == uvec).all())
    self.assertEqual(quantity.symbol, symbol)
    self.assertEqual(quantity.label, label)
    self.assertEqual(quantity.latex, latex)

  def test_independence(self):
    Quantity.participantsAreIndepended = False
    # only add and mul are tested
    x = Quantity(1, 1)
    y = Quantity(1, 1)
    z = Quantity(1, 0)

    mul = lambda x, y: x * y
    add = lambda x, y: x + y


    # this should not raise an error
    try:
      a = x * 3
      a = x * z 
    except:
      self.assertTrue(False)

    # this should raise an error
    self.assertRaises(data.ParticipantsAreNotIndependend, mul, x, x)
    self.assertRaises(data.ParticipantsAreNotIndependend, add, x, x)
    self.assertRaises(data.ParticipantsAreNotIndependend, mul, x, y)
    self.assertRaises(data.ParticipantsAreNotIndependend, add, x, y)

    Quantity.yesIKnowTheDangersAndPromiseToBeCareful()
    # this should not raise an error
    mul(x, x)
    add(x, x)
    mul(x, y)
    add(x, y)

  
  def test_init(self):
    # value and error
    self.assertQuantity(Quantity(),                 1, 0)
    self.assertQuantity(Quantity(6),                6, 0)
    self.assertQuantity(Quantity(6, 3),             6, 3)
    self.assertQuantity(Quantity(6, variance=4),    6, 2)
    self.assertQuantity(Quantity(6, 3, variance=4), 6, 3)

    # naming
    self.assertEqual(Quantity(symbol='fr').symbol, 'fr')
    self.assertEqual(Quantity(label='fr').label,   'fr')
    self.assertEqual(Quantity(latex='fr').latex,   'fr')


    #### unit with value and error
    # iterable
    vec = [0, 1, 2, 3, 4, 5, 6, 7]
    self.assertQuantity(Quantity(7, 2, vec), 7, 2, vec)
    
    # other Quantity object
    x = Quantity(9, 4, vec)
    self.assertQuantity(Quantity(2, 3, x), 2*9, math.sqrt(2**2 * 4**2 + 9**2 * 3**2), vec)

    # float, int
    self.assertQuantity(Quantity(7, 2, 3), 7*3, 2*3)
    self.assertQuantity(Quantity(7, 2, 3.3), 7*3.3, 2*3.3)

    # string
    self.assertQuantity(Quantity(7, 2, ''),               7, 2)
    self.assertQuantity(Quantity(7, 2, '3'),              7*3, 2*3)
    self.assertQuantity(Quantity(7, 2, '3.3'),            7*3.3, 2*3.3)
    self.assertQuantity(Quantity(7, 2, '3.3e-4'),         7*3.3e-4, 2*3.3e-4)
    self.assertQuantity(Quantity(7, 2, 'm'),              7, 2, self.METER)
    self.assertQuantity(Quantity(7, 2, 'm kg'),           7, 2, [1, 0, 1] + [0]*5)
    self.assertQuantity(Quantity(7, 2, 'm * kg'),         7, 2, [1, 0, 1] + [0]*5)
    self.assertQuantity(Quantity(7, 2, '3  * m kg '),     7*3, 2*3, [1, 0, 1] + [0]*5)
    self.assertQuantity(Quantity(7, 2, '3 m^2 kg^-1 s'),  7*3, 2*3, [2, 1, -1] + [0]*5)
    self.assertQuantity(Quantity(7, 2, 'um'),             7e-6, 2e-6, self.METER)
    self.assertQuantity(Quantity(7, 2, 'km'),             7e3, 2e3, self.METER)
    self.assertQuantity(Quantity(7, 2, 'cm^2'),           7e-4, 2e-4, [2] + [0]*7)
    self.assertQuantity(Quantity('5 N m'),                5, 0, [2, -2, 1] + [0]*5)
    self.assertQuantity(Quantity('5+-1 N m'),             5, 1, [2, -2, 1] + [0]*5)
    self.assertQuantity(Quantity('5+-1 4+-2 N m'),        5*4, math.sqrt(5**2 * 2**2+4**2 * 1**2), [2, -2, 1] + [0]*5)
    self.assertQuantity(Quantity(5, 1, '4+-2 N m'),       5*4, math.sqrt(5**2 * 2**2+4**2 * 1**2), [2, -2, 1] + [0]*5)
    self.assertQuantity(Quantity(3, 1, 'g'),              3e-3, 1e-3, [0, 0, 1] + [0]*5)
    self.assertQuantity(Quantity(3, 1, 'mg'),             3e-6, 1e-6, [0, 0, 1] + [0]*5)
    self.assertQuantity(Quantity(3, 1, 'Meter'),          3, 1, self.METER)
    self.assertQuantity(Quantity('4 Meter Second^3'),     4, 0, [1, 3] + [0]*6)
    self.assertQuantity(Quantity('4 Hertz'),              4, 0, [0, -1] + [0]*6)

    # full
    x = Quantity(7, 1, 'Meter', symbol='fs', label='fl', latex='\\mu')
    self.assertQuantity(x, 7, 1, self.METER, 'fs', 'fl', '\\mu')

  def test_unitless(self):
    self.assertFalse(data.Meter.unitless())
    self.assertFalse(data.Second.unitless())
    self.assertFalse(data.Kilogram.unitless())
    self.assertFalse(data.Ampere.unitless())
    self.assertFalse(data.Kelvin.unitless())
    self.assertFalse(data.Mol.unitless())
    self.assertFalse(data.Candela.unitless())
    self.assertTrue(data.Radian.unitless())

  def test_comparison(self):
    # equal
    self.assertTrue(Quantity(2, 1)         == 2)
    self.assertTrue(Quantity(2, 1)         == 1)
    self.assertTrue(Quantity(2, 0, 'm')    == 2)
    self.assertTrue(Quantity(2, 1, 'm')    == 2)
    self.assertTrue(Quantity(2, 1, 'm')    == 1)
    self.assertFalse(Quantity(2, 1, 'm')   == 0)
    self.assertFalse(Quantity(2, 1, 'm')   == 7)
    self.assertFalse(Quantity(2)           == 1)
    self.assertFalse(Quantity(2, 1)        == 0)
    self.assertFalse(Quantity(2, 1)        == 6)

    a = Quantity(3, 0)
    b = Quantity(3, 1)
    c = Quantity(3, 3, 'm')
    self.assertTrue(Quantity(3)            == a)
    self.assertTrue(Quantity(2)            == b)
    self.assertTrue(Quantity(1.6, 1)       == b)
    self.assertFalse(Quantity(1.5, 1)      == b)
    self.assertFalse(Quantity(1)           == a)
    self.assertFalse(Quantity(1.6)         == b)

    eq = lambda x, y: x==y
    self.assertRaises(data.IncompatibleUnits, eq, a, c)

    # greater than
    self.assertTrue(Quantity(2)            > 1)
    self.assertTrue(Quantity(2, 1)         > 0)
    self.assertTrue(Quantity(2, 0, 'm')    > 1)
    self.assertTrue(Quantity(2, 1, 'm')    > 0)
    self.assertFalse(Quantity(2, 1, 'm')   > 6)
    self.assertFalse(Quantity(2, 1)        > 1)
    self.assertFalse(Quantity(2, 1, 'm')   > 1)
    self.assertFalse(Quantity(2, 1)        > 6)

    a = Quantity(3, 0)
    b = Quantity(3, 1)
    c = Quantity(3, 3, 'm')
    self.assertTrue(Quantity(5)            > b)
    self.assertTrue(Quantity(5)            > a)
    self.assertFalse(Quantity(3)           > a)
    self.assertFalse(Quantity(2)           > b)
    self.assertFalse(Quantity(4)           > b)

    eq = lambda x, y: x==y
    self.assertRaises(data.IncompatibleUnits, eq, a, c)

    # less than
    self.assertTrue(Quantity(2)            < 3)
    self.assertTrue(Quantity(2, 1)         < 4)
    self.assertTrue(Quantity(2, 0, 'm')    < 3)
    self.assertTrue(Quantity(2, 1, 'm')    < 4)
    self.assertFalse(Quantity(2, 1, 'm')   < 0)
    self.assertFalse(Quantity(2, 1)        < 3)
    self.assertFalse(Quantity(2, 1, 'm')   < 3)
    self.assertFalse(Quantity(2, 1)        < 0)

    a = Quantity(3, 0)
    b = Quantity(3, 1)
    c = Quantity(3, 3, 'm')
    self.assertTrue(Quantity(2)            < a)
    self.assertTrue(Quantity(1)            < b)
    self.assertFalse(Quantity(3)           < a)
    self.assertFalse(Quantity(2)           < b)
    self.assertFalse(Quantity(4)           < b)

    eq = lambda x, y: x==y
    self.assertRaises(data.IncompatibleUnits, eq, a, c)

    # mixed
    # this will only test the correct linking. More rigorous tests located above.
    a = Quantity(4, 1)
    b = Quantity(3, 1)
    c = Quantity(0, 1)
    self.assertTrue(a != c)
    self.assertTrue(a >= b)
    self.assertTrue(b <= a)
    


  def test_binaryOperators(self):
    # add
    self.assertQuantity(Quantity(4, 3) + 9,                         13, 3)
    self.assertQuantity(Quantity(4, 3) + Quantity(9),               13, 3)
    self.assertQuantity(Quantity(4, 3) + Quantity(9, 4),            13, 5)
    self.assertQuantity(Quantity(4, 3, 'm') + 9,                    13, 3, self.METER)
    self.assertQuantity(Quantity(4, 3, 'm') + Quantity(9, 4, 'm'),  13, 5, self.METER)

    try:
      a = data.Meter + Quantity('m rad')
      a = a + data.Meter
    except data.IncompatibleUnits: self.assertTrue(False)

    add = lambda x, y: x+y
    self.assertRaises(data.IncompatibleUnits, add, data.Meter, data.Second)
    self.assertRaises(data.IncompatibleUnits, add, data.Meter, data.Radian)
    self.assertRaises(data.IncompatibleUnits, add, data.Meter, data.Quantity())

    # sub
    self.assertQuantity(Quantity(4, 3) - 9,                         -5, 3)
    self.assertQuantity(Quantity(4, 3) - Quantity(9),               -5, 3)
    self.assertQuantity(Quantity(4, 3) - Quantity(9, 4),            -5, 5)
    self.assertQuantity(Quantity(4, 3, 'm') - 9,                    -5, 3, self.METER)
    self.assertQuantity(Quantity(4, 3, 'm') - Quantity(9, 4, 'm'),  -5, 5, self.METER)

    try:
      a = data.Meter - Quantity('m rad')
      a = a - data.Meter
    except data.IncompatibleUnits: self.assertTrue(False)

    sub = lambda x, y: x-y
    self.assertRaises(data.IncompatibleUnits, sub, data.Meter, data.Second)
    self.assertRaises(data.IncompatibleUnits, sub, data.Meter, data.Radian)
    self.assertRaises(data.IncompatibleUnits, sub, data.Meter, data.Quantity())

    # mul
    err = math.sqrt(3**2 * 2**2 + 2**2 * 1**2)
    self.assertQuantity(Quantity(3, 1) * 2, 6, 2)
    self.assertQuantity(Quantity(3, 1) * Quantity(2, 2), 6, err)
    self.assertQuantity(Quantity(3, 1, 'm rad') * Quantity(2, 2), 6, err, [1]+[0]*6+[1])
    self.assertQuantity(Quantity(3, 1, 'm rad') * Quantity(2, 2, 's'), 6, err, [1, 1]+[0]*5+[1])
    
    # true div
    err = math.sqrt(1**2 / 2**2 + 2**2 * 2**2)
    self.assertQuantity(Quantity(8, 1) / 2, 4, 0.5)
    self.assertQuantity(Quantity(8, 1) / Quantity(2, 2), 4, err)
    self.assertQuantity(Quantity(8, 1, 'm rad') / Quantity(2, 2), 4, err, [1]+[0]*6+[1])
    self.assertQuantity(Quantity(8, 1, 'm rad') / Quantity(2, 2, 's'), 4, err, [1, -1]+[0]*5+[1])

    # pow
    err = math.sqrt((5*5**4)**2 * 2**2 + (5**5 * math.log(5))**2 * 1)
    self.assertQuantity(Quantity(5, 2)**2, 25, 20)
    self.assertQuantity(Quantity(5, 2, 'm')**2, 25, 20, [2]+[0]*7)
    self.assertQuantity(Quantity(5, 2)**Quantity(2), 25, 20)
    self.assertQuantity(Quantity(5, 2)**Quantity(5, 1), 5**5, err)
    self.assertQuantity(Quantity(5, 2, 'm')**Quantity(5, 1), 5**5, err, [5]+[0]*7)
    self.assertQuantity(Quantity(5, 2, 'm')**Quantity(5, 1, 'rad'), 5**5, err, [5]+[0]*7)

    po = lambda x, y: x**y
    self.assertRaises(data.IncompatibleUnits, po, Quantity(5, 2, 'm'), Quantity(5, 1, 'm'))

    # test r-forms
    # this will only test the correct linking. More rigorous tests located above.
    self.assertQuantity(3 + Quantity(1, 2), 4, 2)
    self.assertQuantity(3 + Quantity(1, 2,'m'), 4, 2, self.METER)
    self.assertQuantity(3 - Quantity(1, 2), 2, 2)
    self.assertQuantity(3 - Quantity(1, 2,'m'), 2, 2, self.METER)
    self.assertQuantity(3 * Quantity(2, 1), 6, 3)
    self.assertQuantity(3 * Quantity(2, 1,'m'), 6, 3, self.METER)
    self.assertQuantity(8 / Quantity(2, 1), 4, 2)
    self.assertQuantity(8 / Quantity(2, 1,'m'), 4, 2, [-1]+[0]*7)
    err = math.log(2) * 2**3 * 1
    self.assertQuantity(2** Quantity(3, 1), 8, err)

    self.assertRaises(data.IncompatibleUnits, po, 8, Quantity(2, 1,'m'))



  def test_errorManipulation(self):
    # relative error
    self.assertQuantity(Quantity(3) % 0.1, 3, 0.3)
    self.assertQuantity(Quantity(3, 1) % 0.1, 3, math.sqrt(0.3**2 + 1**2))
    self.assertQuantity(Quantity(3, 1, 'm') % 0.1, 3, math.sqrt(0.3**2 + 1**2), self.METER)
    self.assertQuantity(Quantity(3, 1, 'm') % Quantity(0.1, 100), 3, math.sqrt(0.3**2 + 1**2), self.METER)
    rel = lambda x, r: x % r
    self.assertRaises(data.IncompatibleUnits, rel, Quantity(3, 1, 'm'), Quantity(3, 2, 'm'))
    

    # absolute error
    self.assertQuantity(Quantity(3) | 0.1, 3, 0.1)
    self.assertQuantity(Quantity(3, 1) | 0.1, 3, math.sqrt(0.1**2 + 1**2))
    self.assertQuantity(Quantity(3, 1, 'm') | 0.1, 3, math.sqrt(0.1**2 + 1**2), self.METER)
    self.assertQuantity(Quantity(3, 1, 'm') | Quantity(0.1, 100, 'm'), 3, math.sqrt(0.1**2 + 1**2), self.METER)
    ab = lambda x, r: x | r
    self.assertRaises(data.IncompatibleUnits, ab, Quantity(3, 1, 'm'), Quantity(3, 2))

    # remove
    self.assertQuantity(Quantity(3, 1, 'm').removeError(), 3, 0, self.METER)


  def test_unitaryOperators(self):
    self.assertQuantity( +Quantity(3, 1, 'm'), 3, 1, self.METER)
    self.assertQuantity( -Quantity(3, 1, 'm'), -3, 1, self.METER)
    self.assertAlmostEqual( abs(Quantity(-3, 1, 'm')), 3, 1, self.METER)

    self.assertAlmostEqual(complex(Quantity(-3.3+2j, 1, 'm')), -3.3+2j)
    self.assertAlmostEqual(float(Quantity(-3.2, 1, 'm')), -3.2)
    self.assertAlmostEqual(int(Quantity(-3.2, 1, 'm')), -3)


  def test_len(self):
    # remember these are the single valued tests
    self.assertEqual(len(Quantity(1, 1, 'm', symbol='a', label='Ab', latex='A_b')), 1)

  def test_buildinStr(self):
    self.assertEqual(str(Quantity(3.1, 1.1, 'm')), '3.1 +- 1.1 m')
    self.assertEqual(str(Quantity(3.1, 1.1, 'J')), '3.1 +- 1.1 m^2 s^-2 kg')
    self.assertEqual(str(Quantity(3.1, 1.1, 'm', symbol='a')), 'a = 3.1 +- 1.1 m')
    self.assertEqual(str(Quantity(3.1, 1.1, 'm', label='Ab')), 'Ab: 3.1 +- 1.1 m')
    self.assertEqual(str(Quantity(3.1, 1.1, 'm', symbol='a', label='Ab')), 'a = 3.1 +- 1.1 m')

  def test_stddev(self):
    self.assertAlmostEqual(Quantity(3, 1).stddev(), 1)
    self.assertAlmostEqual(Quantity(3, 2).stddev(), 2)
    self.assertAlmostEqual(Quantity(3, 3, 'm').stddev(), 3)

  def test_sunit(self):
    self.assertEqual(data.Meter.sunit(), 'm')
    self.assertEqual(data.Volt.sunit(), 'm^2 s^-3 kg A^-1')
    self.assertEqual(data.Joule.sunit(), 'm^2 s^-2 kg')
    self.assertEqual(data.Radian.sunit(), 'rad')
    self.assertEqual(Quantity().sunit(), '')

  def test_repr(self):
    x = Quantity(42, 1, 'm', latex='f')
    y = Quantity(42, 1)
    z = Quantity(42, 1, 'm', symbol='s', label='a', latex='g')
    self.assertQuantity(eval(repr(x)), 42, 1, self.METER, latex='f')
    self.assertQuantity(eval(repr(y)), 42, 1)
    self.assertQuantity(eval(repr(z)), 42, 1, self.METER, symbol='s', label='a', latex='g')


  def test_xxxitem(self):
    x = Quantity(42, 3, 'm', symbol='s', label='f', latex='l_a')
    self.assertQuantity(x[0], 42, 3, self.METER, symbol='s', label='f', latex='l_a')
    self.assertQuantity(x   , 42, 3, self.METER, symbol='s', label='f', latex='l_a')

    x[0] = Quantity(43, 1, 'm')
    self.assertQuantity(x[0], 43, 1, self.METER, symbol='s', label='f', latex='l_a')
    self.assertQuantity(x   , 43, 1, self.METER, symbol='s', label='f', latex='l_a')

    x[0] = 5
    self.assertQuantity(x[0], 5, 0 , self.METER, symbol='s', label='f', latex='l_a')
    self.assertQuantity(x   , 5, 0 , self.METER, symbol='s', label='f', latex='l_a')

    x[:] = 5
    self.assertQuantity(x[0], 5, 0 , self.METER, symbol='s', label='f', latex='l_a')
    self.assertQuantity(x   , 5, 0 , self.METER, symbol='s', label='f', latex='l_a')

    def wu(u):  x[0] = Quantity(1, 0, u)
    def wi(i):  x[i] = 5
    def de(i):  del x[i]
    self.assertRaises(data.IncompatibleUnits, wu, 'J')
    self.assertRaises(IndexError, wi, 2)
    self.assertRaises(ValueError, de, 0)

    ran = 0
    for X in x:
      ran += 1
      self.assertQuantity(X, 5, 0 , self.METER, symbol='s', label='f', latex='l_a')
    self.assertEqual(ran, 1)


  def test_calc(self):
    x = Quantity(7, 3)
    y = Quantity(7, 3, 'm')

    sq = lambda a: a**2
    dr = lambda a: 2*a
    pu = lambda a: list(range(8))

    self.assertQuantity(x.calc(sq, dr),  49, 42)
    self.assertQuantity(x.calc(sq, 2*7), 49, 42)
    self.assertQuantity(x.calc(sq),      49, 42)

    self.assertQuantity(y.calc(sq, dr, reqUnitless=False), 49, 42)
    self.assertQuantity(y.calc(sq, dr, reqUnitless=False, propagateUnit=1),  49, 42, self.METER)
    self.assertQuantity(y.calc(sq, dr, reqUnitless=False, propagateUnit=2),  49, 42, [2]+[0]*7)
    self.assertQuantity(y.calc(sq, dr, reqUnitless=False, propagateUnit=pu), 49, 42, list(range(8)))

  def test_funcs(self):
    pi = math.pi

    # sin family
    self.assertQuantity(data.sin(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.sin(Quantity(pi/2, 2)), 1, 0) 
    self.assertAlmostEqual(data.sin(      pi/2 ),    1) 

    self.assertQuantity(data.sinh(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.sinh(Quantity(1, 2)), math.sinh(1), math.cosh(1)*2) 
    self.assertAlmostEqual(  data.sinh(    1    ), math.sinh(1))

    self.assertQuantity(data.asin(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.asin(Quantity(0.5, 2)), math.asin(0.5), 2/math.sqrt(1-0.5**2)) 
    self.assertAlmostEqual(  data.asin(    0.5  ),   math.asin(0.5))

    self.assertQuantity(data.asinh(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.asinh(Quantity(1, 2)), math.asinh(1), 2/math.sqrt(1+1**2)) 
    self.assertAlmostEqual(  data.asinh(    1    ), math.asinh(1))

    # cos family
    self.assertQuantity(data.cos(Quantity(0, 2)), 1, 0) 
    self.assertQuantity(data.cos(Quantity(pi/2, 2)), 0, -2) 
    self.assertAlmostEqual(  data.cos(    pi/2 ),    0)


    self.assertQuantity(data.cosh(Quantity(0, 2)), 1, 0) 
    self.assertQuantity(data.cosh(Quantity(1, 2)), math.cosh(1), math.sinh(1)*2) 
    self.assertAlmostEqual(  data.cosh(    1    ), math.cosh(1))


    self.assertQuantity(data.acos(Quantity(0, 2)), math.acos(0), -2) 
    self.assertQuantity(data.acos(Quantity(0.5, 2)), math.acos(0.5), -2/math.sqrt(1-0.5**2)) 
    self.assertAlmostEqual(  data.acos(         0.5  ),   math.acos(0.5))


    self.assertQuantity(data.acosh(Quantity(2, 2)), math.acosh(2), 2/math.sqrt(2**2-1))
    self.assertQuantity(data.acosh(Quantity(3, 2)), math.acosh(3), 2/math.sqrt(3**2-1))
    self.assertAlmostEqual(  data.acosh(    3    ), math.acosh(3))


    # tan family
    self.assertQuantity(data.tan(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.tan(Quantity(1, 2)), math.tan(1), 2/math.cos(1)**2) 
    self.assertAlmostEqual(  data.tan(    1    ), math.tan(1))


    self.assertQuantity(data.tanh(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.tanh(Quantity(1, 2)), math.tanh(1), 2/math.cosh(1)**2) 
    self.assertAlmostEqual(  data.tanh(    1    ), math.tanh(1))


    self.assertQuantity(data.atan(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.atan(Quantity(1, 2)), math.atan(1), 2/(1+1**2)) 
    self.assertAlmostEqual(  data.atan(         1    ), math.atan(1))

    self.assertQuantity(data.atan2(Quantity(0, 2), Quantity(1, 0)), 0, 2) 
    self.assertQuantity(data.atan2(Quantity(1, 2), Quantity(1, 0)), math.atan(1), 2/(1+1**2)) 
    self.assertAlmostEqual(  data.atan2(         1, 1    ), math.atan(1))
    self.assertQuantity(data.atan2(Quantity(2, 2), Quantity(-1, 1)), math.atan2(2, -1), math.sqrt(8) / (1+2**2))

    self.assertQuantity(data.atanh(Quantity(0, 2)), 0, 2)
    self.assertQuantity(data.atanh(Quantity(0.5, 2)), math.atanh(0.5), 2/(1-0.5**2))
    self.assertAlmostEqual(  data.atanh(    0.5    ), math.atanh(0.5))

    #misc
    self.assertQuantity(data.sqrt(Quantity(1, 2)), 1, 1)
    self.assertQuantity(data.sqrt(Quantity(4, 2)), 2, 1/2)
    self.assertAlmostEqual(data.sqrt(      4    ), 2)
    self.assertQuantity(data.sqrt(Quantity(1, 2, 'm^2')), 1, 1, self.METER)

    self.assertQuantity(data.exp(Quantity(1, 2)), math.e, 2 * math.e)
    self.assertQuantity(data.exp(Quantity(4, 2)), math.e**4, 2*math.e**4)
    self.assertAlmostEqual(data.exp(      4    ), math.e**4)

    self.assertQuantity(data.log(Quantity(1, 2)), 0, 2)
    self.assertQuantity(data.log(Quantity(4, 2)), math.log(4), 1/2)
    self.assertAlmostEqual(data.log(      4    ), math.log(4))

    self.assertQuantity(data.log2(Quantity(1, 2)), 0, 2/math.log(2))
    self.assertQuantity(data.log2(Quantity(4, 2)), 2, 2/(math.log(2) * 4))
    self.assertAlmostEqual(data.log2(      4    ), 2)

    self.assertQuantity(data.log10(Quantity(1, 2)), 0, 2/math.log(10))
    self.assertQuantity(data.log10(Quantity(100, 2)), 2, 2/(math.log(10) * 100))
    self.assertAlmostEqual(data.log10(      100    ), 2)

class QuantityVectorTest(unittest.TestCase):

  def assertQuantity(self, quantity, v, std, uvec=[0, 0, 0, 0, 0, 0, 0, 0], symbol='', label='', latex=''):
    if hasattr(quantity.value, '__len__'):
      self.assertTrue(np.allclose(quantity.value, v))
    else:
      self.assertAlmostEqual(quantity.value, v)

    if hasattr(quantity.variance, '__len__'):
      self.assertTrue(np.allclose(quantity.variance, std**2))
    else:
      self.assertAlmostEqual(quantity.variance, std**2)

    uvec = np.array(uvec)
    self.assertTrue((quantity.uvec == uvec).all())

    self.assertSequenceEqual(quantity.symbol, symbol)
    self.assertSequenceEqual(quantity.label, label)
    self.assertSequenceEqual(quantity.latex, latex)
  
  def setUp(self):
    self.METER = [1] + [0]*7
    Quantity.yesIKnowTheDangersAndPromiseToBeCareful()

    self.x = np.arange(100)
    self.y = np.sqrt(np.arange(100))
    self.z = np.array( (4, 5, 6, 7) ) * math.e

    self.sx = np.zeros(100) + math.pi
    self.sy = np.cos(self.y)**2
    self.sz = np.array( (4, 5, 6, 7) ) / 13

    self.sym = ['s1', 's2', 's3', 's4']
    self.lab = ['Height', 'Width', 'Depth', 'Thickness']
    self.lat =  ['s_1', 's_2', 's_3', 's_4']

    ms  = data.enumstr('s', 100)


  def test_init(self):
    # value and error
    self.assertQuantity(Quantity(self.x), self.x, 0)
    self.assertQuantity(Quantity(self.x, 3), self.x, 3)
    self.assertQuantity(Quantity(self.x, self.sx), self.x, self.sx)
    self.assertQuantity(Quantity(self.x, variance=self.sx**2), self.x, self.sx)
    self.assertQuantity(Quantity(list(self.x), list(self.sx)), self.x, self.sx)

    # naming
    self.assertEqual(Quantity(symbol=self.sym).symbol, self.sym)
    self.assertEqual(Quantity(label=self.lab).label, self.lab)
    self.assertEqual(Quantity(latex=self.lat).latex, self.lat)


    #### unit with value and error
    # iterable
    vec = [0, 1, 2, 3, 4, 5, 6, 7]
    self.assertQuantity(Quantity(self.y, self.sy, vec), self.y, self.sy, vec)
    
    # other Quantity object
    a = Quantity(self.x, self.sx, vec)
    self.assertQuantity(Quantity(2, 3, a), 2*self.x, np.sqrt(2**2 * self.sx**2 + self.x**2 * 3**2), vec)

    # float, int
    self.assertQuantity(Quantity(self.x, self.sx, 3), self.x*3, self.sx*3)
    self.assertQuantity(Quantity(self.y, self.sy, 3.3), self.y*3.3, self.sy*3.3)

    # string
    self.assertQuantity(Quantity(self.x, self.sx, '3 m^2 kg^-1 s'),  self.x*3, self.sx*3, [2, 1, -1] + [0]*5)
    self.assertQuantity(Quantity(self.x, self.sx, '4+-2 N m'),       self.x*4, math.sqrt(self.x**2 * 2**2+4**2 * self.sx**2), [2, -2, 1] + [0]*5)

    # full
    x = Quantity(self.z, self.sz, 'Meter', symbol=self.sym, label=self.lab, latex=self.lat)
    self.assertQuantity(x, self.z, self.sz, self.METER, self.sym, self.lab, self.lat)

  def test_unitless(self):
    self.assertFalse(data.Meter.unitless())
    self.assertFalse(data.Second.unitless())
    self.assertFalse(data.Kilogram.unitless())
    self.assertFalse(data.Ampere.unitless())
    self.assertFalse(data.Kelvin.unitless())
    self.assertFalse(data.Mol.unitless())
    self.assertFalse(data.Candela.unitless())
    self.assertTrue(data.Radian.unitless())

  def test_comparison(self):
    # equal
    self.assertTrue(Quantity(2, 1)         == 2)
    self.assertTrue(Quantity(2, 1)         == 1)
    self.assertTrue(Quantity(2, 0, 'm')    == 2)
    self.assertTrue(Quantity(2, 1, 'm')    == 2)
    self.assertTrue(Quantity(2, 1, 'm')    == 1)
    self.assertFalse(Quantity(2, 1, 'm')   == 0)
    self.assertFalse(Quantity(2, 1, 'm')   == 7)
    self.assertFalse(Quantity(2)           == 1)
    self.assertFalse(Quantity(2, 1)        == 0)
    self.assertFalse(Quantity(2, 1)        == 6)

    a = Quantity(3, 0)
    b = Quantity(3, 1)
    c = Quantity(3, 3, 'm')
    self.assertTrue(Quantity(3)            == a)
    self.assertTrue(Quantity(2)            == b)
    self.assertTrue(Quantity(1.6, 1)       == b)
    self.assertFalse(Quantity(1.5, 1)      == b)
    self.assertFalse(Quantity(1)           == a)
    self.assertFalse(Quantity(1.6)         == b)

    eq = lambda x, y: x==y
    self.assertRaises(data.IncompatibleUnits, eq, a, c)

    # greater than
    self.assertTrue(Quantity(2)            > 1)
    self.assertTrue(Quantity(2, 1)         > 0)
    self.assertTrue(Quantity(2, 0, 'm')    > 1)
    self.assertTrue(Quantity(2, 1, 'm')    > 0)
    self.assertFalse(Quantity(2, 1, 'm')   > 6)
    self.assertFalse(Quantity(2, 1)        > 1)
    self.assertFalse(Quantity(2, 1, 'm')   > 1)
    self.assertFalse(Quantity(2, 1)        > 6)

    a = Quantity(3, 0)
    b = Quantity(3, 1)
    c = Quantity(3, 3, 'm')
    self.assertTrue(Quantity(5)            > b)
    self.assertTrue(Quantity(5)            > a)
    self.assertFalse(Quantity(3)           > a)
    self.assertFalse(Quantity(2)           > b)
    self.assertFalse(Quantity(4)           > b)

    eq = lambda x, y: x==y
    self.assertRaises(data.IncompatibleUnits, eq, a, c)

    # less than
    self.assertTrue(Quantity(2)            < 3)
    self.assertTrue(Quantity(2, 1)         < 4)
    self.assertTrue(Quantity(2, 0, 'm')    < 3)
    self.assertTrue(Quantity(2, 1, 'm')    < 4)
    self.assertFalse(Quantity(2, 1, 'm')   < 0)
    self.assertFalse(Quantity(2, 1)        < 3)
    self.assertFalse(Quantity(2, 1, 'm')   < 3)
    self.assertFalse(Quantity(2, 1)        < 0)

    a = Quantity(3, 0)
    b = Quantity(3, 1)
    c = Quantity(3, 3, 'm')
    self.assertTrue(Quantity(2)            < a)
    self.assertTrue(Quantity(1)            < b)
    self.assertFalse(Quantity(3)           < a)
    self.assertFalse(Quantity(2)           < b)
    self.assertFalse(Quantity(4)           < b)

    eq = lambda x, y: x==y
    self.assertRaises(data.IncompatibleUnits, eq, a, c)

    # mixed
    # this will only test the correct linking. More rigorous tests located above.
    a = Quantity(4, 1)
    b = Quantity(3, 1)
    c = Quantity(0, 1)
    self.assertTrue(a != c)
    self.assertTrue(a >= b)
    self.assertTrue(b <= a)
    


  def test_binaryOperators(self):
    # add
    self.assertQuantity(Quantity(4, 3) + 9,                         13, 3)
    self.assertQuantity(Quantity(4, 3) + Quantity(9),               13, 3)
    self.assertQuantity(Quantity(4, 3) + Quantity(9, 4),            13, 5)
    self.assertQuantity(Quantity(4, 3, 'm') + 9,                    13, 3, self.METER)
    self.assertQuantity(Quantity(4, 3, 'm') + Quantity(9, 4, 'm'),  13, 5, self.METER)

    try:
      a = data.Meter + Quantity('m rad')
      a = a + data.Meter
    except data.IncompatibleUnits: self.assertTrue(False)

    add = lambda x, y: x+y
    self.assertRaises(data.IncompatibleUnits, add, data.Meter, data.Second)
    self.assertRaises(data.IncompatibleUnits, add, data.Meter, data.Radian)
    self.assertRaises(data.IncompatibleUnits, add, data.Meter, data.Quantity())

    # sub
    self.assertQuantity(Quantity(4, 3) - 9,                         -5, 3)
    self.assertQuantity(Quantity(4, 3) - Quantity(9),               -5, 3)
    self.assertQuantity(Quantity(4, 3) - Quantity(9, 4),            -5, 5)
    self.assertQuantity(Quantity(4, 3, 'm') - 9,                    -5, 3, self.METER)
    self.assertQuantity(Quantity(4, 3, 'm') - Quantity(9, 4, 'm'),  -5, 5, self.METER)

    try:
      a = data.Meter - Quantity('m rad')
      a = a - data.Meter
    except data.IncompatibleUnits: self.assertTrue(False)

    sub = lambda x, y: x-y
    self.assertRaises(data.IncompatibleUnits, sub, data.Meter, data.Second)
    self.assertRaises(data.IncompatibleUnits, sub, data.Meter, data.Radian)
    self.assertRaises(data.IncompatibleUnits, sub, data.Meter, data.Quantity())

    # mul
    err = math.sqrt(3**2 * 2**2 + 2**2 * 1**2)
    self.assertQuantity(Quantity(3, 1) * 2, 6, 2)
    self.assertQuantity(Quantity(3, 1) * Quantity(2, 2), 6, err)
    self.assertQuantity(Quantity(3, 1, 'm rad') * Quantity(2, 2), 6, err, [1]+[0]*6+[1])
    self.assertQuantity(Quantity(3, 1, 'm rad') * Quantity(2, 2, 's'), 6, err, [1, 1]+[0]*5+[1])
    
    # true div
    err = math.sqrt(1**2 / 2**2 + 2**2 * 2**2)
    self.assertQuantity(Quantity(8, 1) / 2, 4, 0.5)
    self.assertQuantity(Quantity(8, 1) / Quantity(2, 2), 4, err)
    self.assertQuantity(Quantity(8, 1, 'm rad') / Quantity(2, 2), 4, err, [1]+[0]*6+[1])
    self.assertQuantity(Quantity(8, 1, 'm rad') / Quantity(2, 2, 's'), 4, err, [1, -1]+[0]*5+[1])

    # pow
    err = math.sqrt((5*5**4)**2 * 2**2 + (5**5 * math.log(5))**2 * 1)
    self.assertQuantity(Quantity(5, 2)**2, 25, 20)
    self.assertQuantity(Quantity(5, 2, 'm')**2, 25, 20, [2]+[0]*7)
    self.assertQuantity(Quantity(5, 2)**Quantity(2), 25, 20)
    self.assertQuantity(Quantity(5, 2)**Quantity(5, 1), 5**5, err)
    self.assertQuantity(Quantity(5, 2, 'm')**Quantity(5, 1), 5**5, err, [5]+[0]*7)
    self.assertQuantity(Quantity(5, 2, 'm')**Quantity(5, 1, 'rad'), 5**5, err, [5]+[0]*7)

    po = lambda x, y: x**y
    self.assertRaises(data.IncompatibleUnits, po, Quantity(5, 2, 'm'), Quantity(5, 1, 'm'))

    # test r-forms
    # this will only test the correct linking. More rigorous tests located above.
    self.assertQuantity(3 + Quantity(1, 2), 4, 2)
    self.assertQuantity(3 + Quantity(1, 2,'m'), 4, 2, self.METER)
    self.assertQuantity(3 - Quantity(1, 2), 2, 2)
    self.assertQuantity(3 - Quantity(1, 2,'m'), 2, 2, self.METER)
    self.assertQuantity(3 * Quantity(2, 1), 6, 3)
    self.assertQuantity(3 * Quantity(2, 1,'m'), 6, 3, self.METER)
    self.assertQuantity(8 / Quantity(2, 1), 4, 2)
    self.assertQuantity(8 / Quantity(2, 1,'m'), 4, 2, [-1]+[0]*7)
    err = math.log(2) * 2**3 * 1
    self.assertQuantity(2** Quantity(3, 1), 8, err)

    self.assertRaises(data.IncompatibleUnits, po, 8, Quantity(2, 1,'m'))



  def test_errorManipulation(self):
    # relative error
    self.assertQuantity(Quantity(3) % 0.1, 3, 0.3)
    self.assertQuantity(Quantity(3, 1) % 0.1, 3, math.sqrt(0.3**2 + 1**2))
    self.assertQuantity(Quantity(3, 1, 'm') % 0.1, 3, math.sqrt(0.3**2 + 1**2), self.METER)
    self.assertQuantity(Quantity(3, 1, 'm') % Quantity(0.1, 100), 3, math.sqrt(0.3**2 + 1**2), self.METER)
    rel = lambda x, r: x % r
    self.assertRaises(data.IncompatibleUnits, rel, Quantity(3, 1, 'm'), Quantity(3, 2, 'm'))
    

    # absolute error
    self.assertQuantity(Quantity(3) | 0.1, 3, 0.1)
    self.assertQuantity(Quantity(3, 1) | 0.1, 3, math.sqrt(0.1**2 + 1**2))
    self.assertQuantity(Quantity(3, 1, 'm') | 0.1, 3, math.sqrt(0.1**2 + 1**2), self.METER)
    self.assertQuantity(Quantity(3, 1, 'm') | Quantity(0.1, 100, 'm'), 3, math.sqrt(0.1**2 + 1**2), self.METER)
    ab = lambda x, r: x | r
    self.assertRaises(data.IncompatibleUnits, ab, Quantity(3, 1, 'm'), Quantity(3, 2))

    # remove
    self.assertQuantity(Quantity(3, 1, 'm').removeError(), 3, 0, self.METER)


  def test_unitaryOperators(self):
    self.assertQuantity( +Quantity(3, 1, 'm'), 3, 1, self.METER)
    self.assertQuantity( -Quantity(3, 1, 'm'), -3, 1, self.METER)
    self.assertAlmostEqual( abs(Quantity(-3, 1, 'm')), 3, 1, self.METER)

    self.assertAlmostEqual(complex(Quantity(-3.3+2j, 1, 'm')), -3.3+2j)
    self.assertAlmostEqual(float(Quantity(-3.2, 1, 'm')), -3.2)
    self.assertAlmostEqual(int(Quantity(-3.2, 1, 'm')), -3)


  def test_len(self):
    # remember these are the single valued tests
    self.assertEqual(len(Quantity(1, 1, 'm', symbol='a', label='Ab', latex='A_b')), 1)

  def test_buildinStr(self):
    self.assertEqual(str(Quantity(3.1, 1.1, 'm')), '3.1 +- 1.1 m')
    self.assertEqual(str(Quantity(3.1, 1.1, 'J')), '3.1 +- 1.1 m^2 s^-2 kg')
    self.assertEqual(str(Quantity(3.1, 1.1, 'm', symbol='a')), 'a = 3.1 +- 1.1 m')
    self.assertEqual(str(Quantity(3.1, 1.1, 'm', label='Ab')), 'Ab: 3.1 +- 1.1 m')
    self.assertEqual(str(Quantity(3.1, 1.1, 'm', symbol='a', label='Ab')), 'a = 3.1 +- 1.1 m')

  def test_stddev(self):
    self.assertAlmostEqual(Quantity(3, 1).stddev(), 1)
    self.assertAlmostEqual(Quantity(3, 2).stddev(), 2)
    self.assertAlmostEqual(Quantity(3, 3, 'm').stddev(), 3)

  def test_sunit(self):
    self.assertEqual(data.Meter.sunit(), 'm')
    self.assertEqual(data.Volt.sunit(), 'm^2 s^-3 kg A^-1')
    self.assertEqual(data.Joule.sunit(), 'm^2 s^-2 kg')
    self.assertEqual(data.Radian.sunit(), 'rad')
    self.assertEqual(Quantity().sunit(), '')

  def test_repr(self):
    x = Quantity(42, 1, 'm', latex='f')
    y = Quantity(42, 1)
    z = Quantity(42, 1, 'm', symbol='s', label='a', latex='g')
    self.assertQuantity(eval(repr(x)), 42, 1, self.METER, latex='f')
    self.assertQuantity(eval(repr(y)), 42, 1)
    self.assertQuantity(eval(repr(z)), 42, 1, self.METER, symbol='s', label='a', latex='g')


  def test_xxxitem(self):
    x = Quantity(42, 3, 'm', symbol='s', label='f', latex='l_a')
    self.assertQuantity(x[0], 42, 3, self.METER, symbol='s', label='f', latex='l_a')
    self.assertQuantity(x   , 42, 3, self.METER, symbol='s', label='f', latex='l_a')

    x[0] = Quantity(43, 1, 'm')
    self.assertQuantity(x[0], 43, 1, self.METER, symbol='s', label='f', latex='l_a')
    self.assertQuantity(x   , 43, 1, self.METER, symbol='s', label='f', latex='l_a')

    x[0] = 5
    self.assertQuantity(x[0], 5, 0 , self.METER, symbol='s', label='f', latex='l_a')
    self.assertQuantity(x   , 5, 0 , self.METER, symbol='s', label='f', latex='l_a')

    x[:] = 5
    self.assertQuantity(x[0], 5, 0 , self.METER, symbol='s', label='f', latex='l_a')
    self.assertQuantity(x   , 5, 0 , self.METER, symbol='s', label='f', latex='l_a')

    def wu(u):  x[0] = Quantity(1, 0, u)
    def wi(i):  x[i] = 5
    def de(i):  del x[i]
    self.assertRaises(data.IncompatibleUnits, wu, 'J')
    self.assertRaises(IndexError, wi, 2)
    self.assertRaises(ValueError, de, 0)

    ran = 0
    for X in x:
      ran += 1
      self.assertQuantity(X, 5, 0 , self.METER, symbol='s', label='f', latex='l_a')
    self.assertEqual(ran, 1)


  def test_calc(self):
    x = Quantity(7, 3)
    y = Quantity(7, 3, 'm')

    sq = lambda a: a**2
    dr = lambda a: 2*a
    pu = lambda a: list(range(8))

    self.assertQuantity(x.calc(sq, dr),  49, 42)
    self.assertQuantity(x.calc(sq, 2*7), 49, 42)
    self.assertQuantity(x.calc(sq),      49, 42)

    self.assertQuantity(y.calc(sq, dr, reqUnitless=False), 49, 42)
    self.assertQuantity(y.calc(sq, dr, reqUnitless=False, propagateUnit=1),  49, 42, self.METER)
    self.assertQuantity(y.calc(sq, dr, reqUnitless=False, propagateUnit=2),  49, 42, [2]+[0]*7)
    self.assertQuantity(y.calc(sq, dr, reqUnitless=False, propagateUnit=pu), 49, 42, list(range(8)))

  def test_funcs(self):
    pi = math.pi

    # sin family
    self.assertQuantity(data.sin(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.sin(Quantity(pi/2, 2)), 1, 0) 
    self.assertAlmostEqual(data.sin(      pi/2 ),    1) 

    self.assertQuantity(data.sinh(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.sinh(Quantity(1, 2)), math.sinh(1), math.cosh(1)*2) 
    self.assertAlmostEqual(  data.sinh(    1    ), math.sinh(1))

    self.assertQuantity(data.asin(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.asin(Quantity(0.5, 2)), math.asin(0.5), 2/math.sqrt(1-0.5**2)) 
    self.assertAlmostEqual(  data.asin(    0.5  ),   math.asin(0.5))

    self.assertQuantity(data.asinh(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.asinh(Quantity(1, 2)), math.asinh(1), 2/math.sqrt(1+1**2)) 
    self.assertAlmostEqual(  data.asinh(    1    ), math.asinh(1))

    # cos family
    self.assertQuantity(data.cos(Quantity(0, 2)), 1, 0) 
    self.assertQuantity(data.cos(Quantity(pi/2, 2)), 0, -2) 
    self.assertAlmostEqual(  data.cos(    pi/2 ),    0)


    self.assertQuantity(data.cosh(Quantity(0, 2)), 1, 0) 
    self.assertQuantity(data.cosh(Quantity(1, 2)), math.cosh(1), math.sinh(1)*2) 
    self.assertAlmostEqual(  data.cosh(    1    ), math.cosh(1))


    self.assertQuantity(data.acos(Quantity(0, 2)), math.acos(0), -2) 
    self.assertQuantity(data.acos(Quantity(0.5, 2)), math.acos(0.5), -2/math.sqrt(1-0.5**2)) 
    self.assertAlmostEqual(  data.acos(         0.5  ),   math.acos(0.5))


    self.assertQuantity(data.acosh(Quantity(2, 2)), math.acosh(2), 2/math.sqrt(2**2-1))
    self.assertQuantity(data.acosh(Quantity(3, 2)), math.acosh(3), 2/math.sqrt(3**2-1))
    self.assertAlmostEqual(  data.acosh(    3    ), math.acosh(3))


    # tan family
    self.assertQuantity(data.tan(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.tan(Quantity(1, 2)), math.tan(1), 2/math.cos(1)**2) 
    self.assertAlmostEqual(  data.tan(    1    ), math.tan(1))


    self.assertQuantity(data.tanh(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.tanh(Quantity(1, 2)), math.tanh(1), 2/math.cosh(1)**2) 
    self.assertAlmostEqual(  data.tanh(    1    ), math.tanh(1))


    self.assertQuantity(data.atan(Quantity(0, 2)), 0, 2) 
    self.assertQuantity(data.atan(Quantity(1, 2)), math.atan(1), 2/(1+1**2)) 
    self.assertAlmostEqual(  data.atan(         1    ), math.atan(1))

    self.assertQuantity(data.atan2(Quantity(0, 2), Quantity(1, 0)), 0, 2) 
    self.assertQuantity(data.atan2(Quantity(1, 2), Quantity(1, 0)), math.atan(1), 2/(1+1**2)) 
    self.assertAlmostEqual(  data.atan2(         1, 1    ), math.atan(1))
    self.assertQuantity(data.atan2(Quantity(2, 2), Quantity(-1, 1)), math.atan2(2, -1), math.sqrt(8) / (1+2**2))

    self.assertQuantity(data.atanh(Quantity(0, 2)), 0, 2)
    self.assertQuantity(data.atanh(Quantity(0.5, 2)), math.atanh(0.5), 2/(1-0.5**2))
    self.assertAlmostEqual(  data.atanh(    0.5    ), math.atanh(0.5))

    #misc
    self.assertQuantity(data.sqrt(Quantity(1, 2)), 1, 1)
    self.assertQuantity(data.sqrt(Quantity(4, 2)), 2, 1/2)
    self.assertAlmostEqual(data.sqrt(      4    ), 2)
    self.assertQuantity(data.sqrt(Quantity(1, 2, 'm^2')), 1, 1, self.METER)

    self.assertQuantity(data.exp(Quantity(1, 2)), math.e, 2 * math.e)
    self.assertQuantity(data.exp(Quantity(4, 2)), math.e**4, 2*math.e**4)
    self.assertAlmostEqual(data.exp(      4    ), math.e**4)

    self.assertQuantity(data.log(Quantity(1, 2)), 0, 2)
    self.assertQuantity(data.log(Quantity(4, 2)), math.log(4), 1/2)
    self.assertAlmostEqual(data.log(      4    ), math.log(4))

    self.assertQuantity(data.log2(Quantity(1, 2)), 0, 2/math.log(2))
    self.assertQuantity(data.log2(Quantity(4, 2)), 2, 2/(math.log(2) * 4))
    self.assertAlmostEqual(data.log2(      4    ), 2)

    self.assertQuantity(data.log10(Quantity(1, 2)), 0, 2/math.log(10))
    self.assertQuantity(data.log10(Quantity(100, 2)), 2, 2/(math.log(10) * 100))
    self.assertAlmostEqual(data.log10(      100    ), 2)

if __name__ == '__main__':
  unittest.main()
