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

import casadi as cs
import casadi_extras as ct
import numpy as np
from omsf import util

from .sensory_model import NullModel
from .signal import INERTIAL_SIGNAL, REFERENCE_INERTIAL_SIGNAL, VISUAL_SIGNAL
from .gravity import GRAVITY, DEFAULT_GRAVITY
from casadi_extras import ImplicitDae
import omsf.transform as transform


class Scenario(object):
    '''Simulation scenario.

    Defines all symbolic variables and expressions needed to formulate the optimization problem,
    including motion platform, motion sensor, maneuver(s) and the cost function.
    '''

    def __init__(self, platform, lagrange_term, sensor=NullModel()):
        '''Constructor
        '''

        self._platform = platform
        self._sensor = sensor
        self._lagrangeTerm = lagrange_term
        self.gravity = DEFAULT_GRAVITY
        self.headToPlatform = transform.identity()
        
        x = ct.inequality.vertcat({
            'platform': platform.state,
            'sensor': sensor.state
        })

        z = ct.inequality.vertcat({
            'platform': platform.algState,
            'sensor': sensor.algState,
            'y_in': ct.Inequality(
                expr=INERTIAL_SIGNAL, 
                nominal=INERTIAL_SIGNAL({
                    'f': DEFAULT_GRAVITY,
                    'omega': np.zeros(3),
                    'alpha': np.zeros(3)
                })
            )
        })

        self.state = x
        self.algState = z
        self.input = platform.input
        self.param = platform.param
        self.constraint = platform.constraint

        # Symbolic variable representing initial state, for terminal constraints.
        self.initialState = ct.struct_symMX(x.expr)

        # Symbolic variable representing finial state, for terminal constraints.
        self.finalState = ct.struct_symMX(x.expr)

        # Terminal constraints; can be overriden by user.
        self.terminalConstraint = ct.Inequality()


    def simulate(self, system_trajectory, param=None):
        '''Simulate the scenario for given trajectory and parameters

        TODO: move somewhere else?
        '''

        if param is None:
            param = self.param.nominal

        f_out = self._platform.output
        state = system_trajectory.platformState
        alg_state = system_trajectory.platformAlgState
        input = system_trajectory.input
        time = system_trajectory.time
        
        # TODO: take paramValue from the u, which in the scenario optimization result and hence can also hold the (optimized) parameter values.
        ves = [f_out(
                x=state(t), 
                z=alg_state(t), u=u,
                p=param, g=self.gravity, T_HP=self.headToPlatform)['y']
                for t, u in zip(time[: -1], input(time[: -1]).T)]
        
        return cs.horzcat(*ves)

        '''
        f_out = self._platformOutputFunction.map('f_out_parallel', 'serial', Nt, [3, 4, 5], [])
        ves = f_out(x=u.state[:, : Nt], z=u.algState[:, : Nt], u=u.input[:, : Nt], p=self._motionPlatform.paramValue, g=self.gravity,
            T_HP=self.headToPlatform)['y']
        
        return np.array(ves)
        '''


    def makeDae(self):
        """Make a DAE model.
        """

        x = self.state.expr
        z = self.algState.expr
        u = self.input.expr
        p = self.param.expr
        t = cs.MX.sym('t')
        tdp = ct.struct_MX([
            ct.entry('y_ref', expr=REFERENCE_INERTIAL_SIGNAL),
            ct.entry('u_vis', expr=VISUAL_SIGNAL)
        ])

        [dae_platform] = cs.substitute([self._platform.dae], [GRAVITY], [self.gravity])

        # TODO: replace with a call to platform.evaluateOutput()
        # y = self._motionPlatform.evaluateOutput(gravity=self.gravity, head_to_platform=self.headToPlatform)
        # TODO: add xdot to output function arguments.
        y = self._platform.output(x=self._platform.state.expr, z=self._platform.algState.expr,
            u=self._platform.input.expr, p=self._platform.param.expr, g=self.gravity, T_HP=self.headToPlatform)['y']
        
        dae_sensor = self._sensor.dae

        xdot = ct.struct_MX([
            ct.entry("platform", expr=self._platform.stateDerivative),
            ct.entry("sensor", expr=self._sensor.stateDerivative)
        ])
            
        dae = ct.struct_MX([
            ct.entry('platform', expr=dae_platform),
            ct.entry('sensor', expr=dae_sensor),
            ct.entry('y_in', expr=z['y_in'] - y)
        ])

        # Lagrange term
        L = self._lagrangeTerm

        # DAE model
        return ImplicitDae(xdot=xdot, x=x, z=z, u=u, p=p, t=t, dae=dae, quad=L, tdp=tdp)
