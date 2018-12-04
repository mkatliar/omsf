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
'''
@author: mkatliar
'''
import casadi as cs
import casadi_extras as ct
from omsf import util

import numpy as np


def _interpolate(t, t_base, x):
    return util.interpolate(t, t_base, x)


class SensorySignal:
    ''' Defines visual and inertial signals as continuous functions of time.
    '''

    def __init__(self, t, v_v, omega_v, g_v, f_i, omega_i, alpha_i = None):
        assert t.ndim == 1
        Nt = len(t)
        
        assert v_v.shape     == (3, Nt)
        assert omega_v.shape == (3, Nt)
        assert g_v.shape     == (3, Nt)
        assert f_i.shape     == (3, Nt)
        assert omega_i.shape == (3, Nt)
        
        if alpha_i is None:
            alpha_i = util.time_derivative(omega_i, t)
            
        assert alpha_i.shape == (3, Nt)
        
        self.v_v     = v_v
        self.omega_v = omega_v
        self.g_v     = g_v
        self.f_i     = f_i
        self.omega_i = omega_i
        self.alpha_i = alpha_i
        self.time    = t

        
    @property    
    def startTime(self):
        return self.time[0]

    
    @property    
    def stopTime(self):
        return self.time[-1]

            
    def visualSignal(self, t):
        ''' Get visual signal input for time t.
        '''
        return cs.vertcat(_interpolate(t, self.time, self.v_v),
                          _interpolate(t, self.time, self.omega_v),
                          _interpolate(t, self.time, self.g_v))

        '''
        # Visual input
        u_vis = vectorInterpolant('u_vis', interpolation_method, [sensory_signal.time], 
            np.vstack((sensory_signal.v_v, sensory_signal.omega_v, sensory_signal.g_v)))
        '''

        
    def inertialSignal(self, t):
        ''' Get inertial signal input for time t.
        '''
        return cs.vertcat(_interpolate(t, self.time, self.f_i),
                          _interpolate(t, self.time, self.omega_i),
                          _interpolate(t, self.time, self.alpha_i))

        '''
        # Reference output
        interpolation_method = 'bspline'
        y_ref = vectorInterpolant('y_ref', interpolation_method, [sensory_signal.time], 
            np.vstack((sensory_signal.f_i, sensory_signal.omega_i, sensory_signal.alpha_i)))
        '''

    
    def visualLinearVelocity(self, t):
        ''' Get visual linear velocity for time t.
        '''
        return _interpolate(t, self.time, self.v_v)

    
    def visualAngularVelocity(self, t):
        ''' Get visual angular velocity for time t.
        '''
        return _interpolate(t, self.time, self.omega_v)

    
    def visualDirectionOfGravity(self, t):
        ''' Get visual direction of gravity for time t.
        '''
        return _interpolate(t, self.time, self.g_v)

    
    def inertialSpecificForce(self, t):
        ''' Get specific force for time t.
        '''
        return _interpolate(t, self.time, self.f_i)

    
    def inertialAngularVelocity(self, t):
        ''' Get inertial angular velocity for time t.
        '''
        return _interpolate(t, self.time, self.omega_i)
        
    
    def cut(self, t1, t2):
        ''' Cut sensory input between t1 and t2
        '''
        ind = np.logical_and(self.time >= t1, self.time <= t2)
        return SensorySignal(t = self.time[ind], v_v = self.v_v[:, ind], omega_v = self.omega_v[:, ind], g_v = self.g_v[:, ind],
                            f_i = self.f_i[:, ind], omega_i = self.omega_i[:, ind], alpha_i = self.alpha_i[:, ind])
