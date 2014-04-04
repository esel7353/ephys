#!/usr/bin/python3

import ephys as ep

x = ep.Quantity('mm').name(label='Positon', latex='x_0')
U = ep.Quantity('mV').name(label='Voltage under Neadle', latex='U_N')

ep.daq('small.csv', x, U, ep.StdDev(U), skip=1)
x.variance = 0.00001**2

p = ep.Plot()
p.error(x, U)

offset = ep.Quantity('V').name(symbol='U_0')
m      = ep.Quantity('V/m').name(symbol='m')
lin = ep.PolynomFit(offset, m)

lin.fit(x, U)
p.fit(lin)

p.save('small.png')
