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
#from monopod import DoubleIntegratorMonopod as Monopod
import matplotlib.pyplot as plt
import numpy, casadi as cs, omsf, math

''' Motion platform '''
# Set the head-to-platform matrix so that the Y_head = -Z_platform and Z_head = Y_platform
platform = Monopod(head_to_platform = omsf.transform.rotationX(-math.pi / 2))

# Calculate initial algebraic state
x0 = platform.state()
x0['q'] = 1
x0['v'] = 0
print('q0 = {0}'.format(x0['q']))

z0 = platform.algState()

# Calculate output at initial state.
y0 = platform.output(x = x0, z = z0, p = numpy.array([0]))['o0']

gravity = omsf.DEFAULT_GRAVITY
print('g = {0}'.format(gravity))
print('platform.headToPlatform = {0}'.format(platform.headToPlatform))
print('g_head = {0}'.format(platform.headFrameGravity(x0, z0, gravity)))
print('y0 = {0}'.format(y0))

''' Initialize MDA object ''' 
scenario = omsf.Scenario(platform)
scenario.initialPlatformStateGuess = x0
scenario.initialPlatformAlgStateGuess = z0
scenario.initialVelocityZero = False
scenario.finalVelocityZero   = False
scenario.initialPositionUpright = False
scenario.finalPositionUpright   = False
scenario.controlInputPenalty  = 0.1
#mda.platform = motion.MotionPlatformX()
scenario.optimizationOptions['max_iter'] = 100
scenario.optimizationOptions['linear_solver'] = 'ma86'
#mda.optimizationOptions['tol'] = 1e-7
#mdaobj.optimizationOptions["iteration_callback"] = iteration_callback
scenario.useMaxNorm = False

''' Initialize cost function '''
rotationalVelocityWeight = 10
w = cs.diag([1, 1, 1, rotationalVelocityWeight, rotationalVelocityWeight, rotationalVelocityWeight]) 
scenario.lagrangeTerm = omsf.cost_euclidean(w)

''' Check initial algebraic state '''
dae_out0 = platform.daeFunction(x = x0, z = z0, p = numpy.zeros(platform.input.size))
print('dae_out0 = {0}'.format(dae_out0))

''' Load motion '''
data = omsf.RecordedMotion.LoadFromMat('data/AccDec.mat')
si = omsf.recordedMotionToSensorySignal(data, numpy.array([0, 0, -1]))

#for i in range(len(si.time)):
#    si.vestibularInput[:, i] = numpy.array([0, 0, -9.81, 0, 0, 0])
#si.PlotVestibular(plt.figure())
#plt.show()

''' Run the MDA '''
[motion] = scenario.optimize([si])
omsf.evaluateMotion(plt.figure(), si, motion, scenario.platform)

plt.figure()
plt.plot(motion.time[: -1], motion.input.T)
plt.xlabel('time')
plt.ylabel('u')
plt.grid(True)

plt.figure()
plt.plot(motion.time, motion.state.T)
plt.xlabel('time')
plt.ylabel('x')
plt.grid(True)

plt.show()