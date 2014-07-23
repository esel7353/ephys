#!/usr/bin/python3

import ephys as ep

t = ep.Quantity(0, 0.0001, 's').name(label='Time', latex='t')
U = ep.Quantity(0, 0.02, 'V').name(label='Photo Diode Voltage', latex='U_{PD}')


ep.daq('data.csv', t, U, skip=2)
t.prefunit('ms')

p = ep.Plot()
p.error(t, U, ecolor='0.3')

A     = ep.Quantity('V').name(latex='U_0')
mu    = ep.Quantity('s').name(latex='\\mu')
gamma = ep.Quantity('s').name(latex='\\Gamma')

mf = ep.LorentzFit(A, mu, gamma)  

mf.fit(t, U)
p.fit(mf)
p.save('tut.png')
