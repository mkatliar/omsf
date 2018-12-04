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
import casadi as cs
import casadi_extras as ct
import numpy as np

from .motion_platform import MotionPlatform
from .gravity import GRAVITY
from .util import inertialSignalFromTransformationMatrix


class DoubleIntegratorMotionPlatform(MotionPlatform):

    def __init__(self, q, T, motion_limits, nominal_position, param=ct.BoundedVariable(), **kwargs):
        '''Constructor
        '''

        v = cs.MX.sym('v', q.shape)
        u = cs.MX.sym('u', q.shape)

        x = ct.struct_MX([
            ct.entry('q', expr=q),
            ct.entry('v', expr=v)
        ])
        
        # Init DAE function
        xdot = ct.struct_symMX(x)
        dae = cs.vertcat(
            xdot['q'] - v,
            xdot['v'] - u
        )
        
        # Create f_P, omega_P and alpha_P from T
        R_WP = cs.transpose(T[: 3, : 3])
        a_P, omega_P, alpha_P = inertialSignalFromTransformationMatrix(T, q, v, u)
        f_P = cs.mtimes(R_WP, GRAVITY) - a_P
        T_PW = T

        # Axes limits
        q_min = np.array([l.positionMin for l in motion_limits])
        q_max = np.array([l.positionMax for l in motion_limits])
        v_min = np.array([l.velocityMin for l in motion_limits])
        v_max = np.array([l.velocityMax for l in motion_limits])
        u_min = np.array([l.accelerationMin for l in motion_limits])
        u_max = np.array([l.accelerationMax for l in motion_limits])
        q_min_finite, = np.where(q_min > -np.Inf)
        q_max_finite, = np.where(q_max <  np.Inf)

        # Init constraint expression
        constr = ct.struct_MX([
            ct.entry("cq1", expr = v[q_max_finite] ** 2 + 2 * (q_max[q_max_finite] - q[q_max_finite]) * u_min[q_max_finite]),
            ct.entry("cq2", expr = v[q_min_finite] ** 2 + 2 * (q_min[q_min_finite] - q[q_min_finite]) * u_max[q_min_finite])
        ])

        # Init state constraint bounds
        lbg = constr()
        ubg = constr()
        lbg["cq1"] = -np.Inf
        ubg["cq1"] = 0        
        lbg["cq2"] = -np.Inf
        ubg["cq2"] = 0

        # Init base class
        # TODO: pass input and state bounds together as a dict similar to param and constr?
        MotionPlatform.__init__(self, state_derivative=xdot, 
            state=ct.Inequality(
                expr=x, 
                lb=x({'q': q_min, 'v': v_min}),
                ub=x({'q': q_max, 'v': v_max}),
                nominal=x({'q': nominal_position, 'v': 0})
            ), 
            input=ct.Inequality(
                expr=u,
                lb=u_min,
                ub=u_max,
                nominal=cs.DM.zeros(u.numel())
            ), 
            dae=dae, f_P=f_P, omega_P=omega_P, alpha_P=alpha_P, T_PW=T_PW,
            constraint=ct.Inequality(
                expr=constr, 
                lb=lbg, 
                ub=ubg
            ),
            param=param)
        
        self._axesLimits = motion_limits

        
    @property
    def numberOfAxes(self):
        return len(self._axesLimits)

        
    @property
    def axesLimits(self):
        return self._axesLimits


    @staticmethod
    def _makeVariables(n_axes):
        x = ct.struct_symMX([
            ct.entry("q", shape = n_axes), 
            ct.entry("v", shape = n_axes)
        ])
        
        z = ct.struct_symMX([
            ct.entry("empty", shape = 0)
        ])
        
        u = ct.struct_symMX([
            ct.entry("u", shape = n_axes)
        ])
        
        return x, z, u

    
    def HeadToWorld(self, x, z):
        '''TODO: remove?
        '''
        return cs.mtimes(self._PlatformToWorld(self.state(x.cat)['q']), self.headToPlatform)
