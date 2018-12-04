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
u = numpy.array([1])

alpha0 = numpy.pi / 3    # Since x0[0] = norm(r1) = norm(r0)

z0 = platform.algState()
z0['R'] = numpy.array([[numpy.cos(alpha0), -numpy.sin(alpha0)], [numpy.sin(alpha0), numpy.cos(alpha0)]]) # Rotation matrix corresponding to alpha0
z0['Rdot']  = numpy.zeros((2, 2))   # Zero, since the velocity x0[1] is 0
z0['Rddot'] = numpy.zeros((2, 2))   # Zero, since the acceleration u is 0

integrator = cs.Integrator('integrator', 'idas', platform.daeFunction, {'tf' : 0.1})
print(integrator)

out = integrator(x0 = x0, z0 = z0, p = u)

J_xf_z0 = integrator.jacobian(2, 0)
J_zf_z0 = integrator.jacobian(2, 2)
J_xf_x0 = integrator.jacobian(0, 0)
J_zf_x0 = integrator.jacobian(0, 2)

print('Sensitivity dxf/dz0 = {0}'.format(J_xf_z0(x0 = x0, z0 = z0, p = u)['jac']))
print('Sensitivity dzf/dz0 = {0}'.format(J_zf_z0(x0 = x0, z0 = z0, p = u)['jac']))
print('Sensitivity dxf/dx0 = {0}'.format(J_xf_x0(x0 = x0, z0 = z0, p = u)['jac']))
print('Sensitivity dzf/dx0 = {0}'.format(J_zf_x0(x0 = x0, z0 = z0, p = u)['jac']))

print(out)