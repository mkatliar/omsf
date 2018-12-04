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
'''Motion optimization with a sensory model.

@author: mkatliar
'''

import casadi as cs, numpy, matplotlib.pyplot as plt
import omsf
import omsf.sensory_model
import omsf.io.matlab
from surge_platform import SurgePlatform


platform = SurgePlatform()
sensor = omsf.sensory_model.OrmsbyModel()

#----------------------------------
# The cost function.
#----------------------------------
w = cs.diag([1, 1, 1, 10, 10, 10, 0, 0, 0]) # Weighting matrix
W = cs.mtimes(w.T, w)
delta_y = omsf.INERTIAL_SIGNAL.cat - omsf.REFERENCE_INERTIAL_SIGNAL.cat

L = 0.01 * cs.sumsqr(platform.input.expr) \
    + 0.01 * cs.sumsqr(platform.state.expr) \
    + cs.mtimes([cs.transpose(delta_y), W, delta_y])

scenario = omsf.Scenario(platform=platform, sensor=sensor, lagrange_term=L)

scenario.initialPlatformStateGuess = platform.state.expr(0)
scenario.initialInputGuess = 0

optimizer = omsf.Optimizer()
optimizer.optimizationOptions['ipopt'] = {'linear_solver' : 'ma86'}
#optimizer.optimizationOptions['ipopt']['tol'] = 1e-5
#optimizer.optimizationOptions['ipopt']["iteration_callback"] = iteration_callback
optimizer.jit = False
optimizer.parallelization = 'serial'

data = omsf.io.matlab.loadRecordedMotion('examples/data/AccDec.mat')
si = omsf.recordedMotionToSensorySignal(data, numpy.array([0, 0, -1]))

res = optimizer.optimize(scenario, [si])
[pm] = res['trajectory']
param = res['param']
# If you want to save the result:
# pm.SaveToMat('result.mat')
print('Initial and final platform state: {0}'.format(pm.platformState(pm.time[[0, -1]])))
print('Initial and final sensor state: {0}'.format(pm.sensorState(pm.time[[0, -1]])))

g = [platform.headFrameGravity(
    x=pm.platformState(t), 
    z=pm.platformAlgState(t),
    p=param,
    g=scenario.gravity, T_HP=scenario.headToPlatform)['g'] for t in pm.time[[0, -1]]]

print('Initial and final gravity in head frame: {0}'.format(cs.horzcat(*g)))
print('Optimal parameter value: {0}'.format(param))
print('Objective value: {0}'.format(res['objective']))

omsf.evaluateMotion(plt.figure(), si, pm, param, scenario)    
plt.show()
