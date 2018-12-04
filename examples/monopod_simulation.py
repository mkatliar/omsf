##
## Copyright (c) 2015-2018 Mikhail Katliar, Max Planck Institute for Biological Cybernetics.
## 
## This file is part of Offline Motion Simulation Framework (OMSF) 
## (see https://github.com/mkatliar/omsf).
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.
##
# -*- coding: utf-8 -*-
"""
@author: mkatliar
"""
from monopod import Monopod
import matplotlib.pyplot as plt
import numpy, casadi as cs, omsf
    
platform = Monopod()
#print(platform.outputFunction)

x0 = numpy.array([1, 0])
t = numpy.linspace(0, 1, 10)
u = numpy.array([1])

alpha0 = numpy.pi / 3    # Since x0[0] = norm(r1) = norm(r0)

z0 = platform.algState()

integrator = cs.Integrator('integrator', 'cvodes', platform.daeFunction)
#integrator.setOption('tf', t)

#simulator = casadi.Simulator(integrator, platform.outputFunction, t)
simulator = cs.Simulator('simulator', integrator, t)

out = simulator(x0 = x0, z0 = z0, p = u)
print('State: {0}'.format(out['xf'].T))
print('Alg. state: {0}'.format(out['zf'].T))

y = cs.DMatrix(6, len(t))
for i in range(len(t)):
    y[:, i] = platform.output(x = out['xf'][:, i], z = out['zf'][:, i])['o0']

plt.figure()
plt.plot(t, out['xf'].T)
plt.xlabel('time')
plt.ylabel('state')
plt.grid(True)

omsf.plot_vestibular_input(plt.figure(), t, y)

plt.show()