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
from typing import NamedTuple

import casadi as cs
import casadi_extras as ct
import numpy as np

from .signal import INERTIAL_SIGNAL
from .gravity import GRAVITY
from omsf import transform


'''
class MotionPlatform(NamedTuple):
    x : cs.MX = cs.MX.sym('x', 0)
    z : cs.MX = cs.MX.sym('z', 0)
    u : cs.MX = cs.MX.sym('u', 0)
    ode : cs.MX = cs.MX.sym('ode', 0)
    alg : cs.MX = cs.MX.sym('alg', 0)
'''

class MotionPlatform(object):
    '''Defines dynamics and output quantities of a motion platform.

    @param constr Inequality object defining platform constraints.
    '''

    def __init__(self, state_derivative, state, input, dae, f_P, omega_P, alpha_P, T_PW, 
        alg_state=ct.BoundedVariable(), param=ct.BoundedVariable(), constraint=ct.Inequality()):
        '''Constructor
        '''
        
        self.stateDerivative = state_derivative
        self.state    = state
        self.algState = alg_state
        self.input    = input
        
        # Parameters
        self.param = param
        
        # Implicit DAE equations
        self.dae = dae

        # Expression for specific force in PF
        self.f_P = f_P
        
        # Expression for rotational velocity in PF
        self.omega_P = omega_P
        
        # Expression for rotational acceleration in PF
        self.alpha_P = alpha_P
        
        # Expression for PF->WF transformation matrix
        self.T_PW = T_PW

		# Constraints
        self.constraint = constraint

        # ------------------------------
        # Init platform output function.
        # ------------------------------
        # TODO: use transformInertialSignal()?
        
        # Recalculate to HF
        T_HP = cs.MX.sym('T_HP', 4, 4)
        T_PH = transform.inv(T_HP)
        r_HP = T_HP[: 3, 3]
        R_PH = T_PH[: 3, : 3]
        
        f_H = cs.mtimes(R_PH, f_P - cs.cross(alpha_P, r_HP) - cs.cross(omega_P, cs.cross(omega_P, r_HP)))
        omega_H = cs.mtimes(R_PH, omega_P)
        alpha_H = cs.mtimes(R_PH, alpha_P)
        
        y = ct.struct_MX(INERTIAL_SIGNAL)
        y['f'] = cs.densify(f_H)
        y['omega'] = cs.densify(omega_H)
        y['alpha'] = cs.densify(alpha_H)

        self.output = cs.Function('MotionPlatformOutputFunction',
            [state.expr, alg_state.expr, input.expr, param.expr, GRAVITY, T_HP], [y], 
            ['x', 'z', 'u', 'p', 'g', 'T_HP'], ['y'])

        # Function to compute gravity vector in head frame.
        T_HW = cs.mtimes(T_PW, T_HP)
        self.headFrameGravity = cs.Function('MotionPlatform_headFrameGravity', 
                                                [state.expr, alg_state.expr, param.expr, GRAVITY, T_HP], 
                                                [cs.mtimes(T_HW[: 3, : 3].T, GRAVITY)],
                                                ['x', 'z', 'p', 'g', 'T_HP'], ['g'])

    """
    def headFrameGravity(self, x, z):
        '''Calculate gravity in head frame.
        '''
        return self._headFrameGravityFunction(x, z, self.gravity, self.headToPlatform)
    """


    """
    def evaluateOutput(self, state=None, alg_state=None, input=None, param=None, gravity=None, head_to_platform=None):
        '''Evaluate inertial signal
        '''

        if state is None:
            state = self.state

        if alg_state is None:
            alg_state = self.algState

        if input is None:
            input = self.input

        if param is None:
            param = self.param

        if head_to_platform is None:
            pass    # TODO: self.headToPlatform

        if gravity is None:
            pass    # TODO: self.gravity
            
        return self.output(x=state, z=alg_state, u=input, p=param, g=gravity, T_HP=head_to_platform)['y']
    """