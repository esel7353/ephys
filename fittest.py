# coding: utf-8
from analysis import *
import random
m = Quantity().name(symbol='m')
c = Quantity().name(symbol='c')
def lin(x, m,c):
    return m*x+c
mf = ModelFit(m, c, func=lin, modeq='U = m \cdot I + c')
x = []
y = []
for i in range(100):
  x.append(random.gauss(i, 0.5))
  y.append(random.gauss(2.34*i, 3))

x = Quantity(x, 0.5, 'mA').name(label='Strom', symbol='I')
y = Quantity(y, 2, 'V').name(label='Spannung', symbol='U')

mf.fit(x,y)
mf.plot().show()
