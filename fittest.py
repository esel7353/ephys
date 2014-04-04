#!/usr/bin/python3

# coding: utf-8
from analysis import *
import random
import numpy as np

x = np.arange(0, 200)
y = 50.66 * np.exp(-0.5 * (123-x)**2 / 5**2)

for i in range(len(x)):
  y[i] += random.gauss(0, 0.144)

#
x = Quantity(x, 1).name(label='Channel', symbol='x')
y = Quantity(y, 0.1, '1/ms').name(label='Rate', symbol='R')

A  = Quantity('1/s').name(symbol='A')
mu = Quantity('').name(latex='\\mu')
s  = Quantity('').name(latex='\\sigma')

mf = GaussFit(A, mu, s)

p = Plot()
p.points(x, y)

p.fit( mf.fit(x, y) )
#p.fit( mf.estimate(x,y))





p.save('asdf.png')
