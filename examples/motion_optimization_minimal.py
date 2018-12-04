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
'''Minimal working example of motion optimization.

Created on Sep 28, 2015

@author: mkatliar
'''
import casadi as cs, numpy, matplotlib.pyplot as plt
import omsf
import omsf.io.matlab
from surge_platform import SurgePlatform

import time


plt.ion()


class IterationCallback(object):
    
    def __init__(self, sensory_signal, fig_traj, fig_cost):
        self._sensorySignal = sensory_signal
        self._figTraj = fig_traj
        self._figCost = fig_cost
            

    def __call__(self, traj, param, f):
        for pm, ss in zip(traj, self._sensorySignal):
            state = pm.platformState
            alg_state = pm.platformAlgState
            time = pm.time
            
            omsf.evaluateMotion(self._figTraj, ss, pm, param, scenario=None)

            t, cost = pm.cost.discretize(0.01)
            self._figCost.gca().plot(t, cost.T)
            self._figCost.gca().set_xlabel('time')
            self._figCost.gca().set_ylabel('incongurence')
            self._figCost.gca().grid(True)

        plt.draw()


platform = SurgePlatform()

#----------------------------------
# The cost function.
#----------------------------------

w = cs.diag([1, 1, 1, 10, 10, 10, 0, 0, 0]) # Weighting matrix
W = cs.mtimes(w.T, w)
delta_y = omsf.INERTIAL_SIGNAL.cat - omsf.REFERENCE_INERTIAL_SIGNAL.cat

L = 0.01 * cs.sumsqr(platform.input.expr) \
    + 0.01 * cs.sumsqr(platform.state.expr) \
    + cs.mtimes([cs.transpose(delta_y), W, delta_y])

scenario = omsf.Scenario(platform=platform, lagrange_term=L)

optimizer = omsf.Optimizer()
optimizer.optimizationOptions['ipopt'] = {'linear_solver' : 'ma86'}
#mdaobj.optimizationOptions['ipopt']['tol'] = 1e-5
#mdaobj.optimizationOptions['ipopt']["iteration_callback"] = iteration_callback
optimizer.jit = False
optimizer.parallelization = 'serial'

filename = ['AccDec.mat', 'LaneChange.mat']
data = [omsf.io.matlab.loadRecordedMotion('examples/data/{0}'.format(fn)) for fn in filename]
sensory_signal = [omsf.recordedMotionToSensorySignal(d, numpy.array([0, 0, -1])) for d in data]

fig_traj = plt.figure()
fig_cost = plt.figure()    
optimizer.iterationCallback = IterationCallback(sensory_signal=sensory_signal,
    fig_traj=fig_traj, fig_cost=fig_cost)

t_begin = time.time()
result = optimizer.optimize(scenario, sensory_signal)
t_end = time.time()

param = result['param']

for pm, ss in zip(result['trajectory'], sensory_signal):
    state = pm.platformState
    alg_state = pm.platformAlgState
    time = pm.time
    # If you want to save the result:
    # pm.SaveToMat('result.mat')
    print('Initial and final state: {0}'.format(state(time[[0, -1]])))

    g0 = platform.headFrameGravity(x=state(time[0]), z=alg_state(time[0]), p=param, g=scenario.gravity, T_HP=scenario.headToPlatform)['g']
    g1 = platform.headFrameGravity(x=state(time[-1]), z=alg_state(time[-1]), p=param, g=scenario.gravity, T_HP=scenario.headToPlatform)['g']
    print('Initial and final gravity in head frame: {0}'.format(cs.horzcat(g0, g1)))

    omsf.evaluateMotion(fig_traj, ss, pm, param, scenario)

    t, cost = pm.cost.discretize(0.01)
    fig_cost.gca().plot(t, cost.T)
    fig_cost.gca().set_xlabel('time')
    fig_cost.gca().set_ylabel('incongurence')
    fig_cost.gca().grid(True)

print('Optimal parameter value: {0}'.format(param))
print('Objective value: {0}'.format(result['objective']))
print('Elapsed time: {0}s'.format(t_end - t_begin))

plt.show()
