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
import omsf, casadi as cs, casadi.tools as ct, numpy, math

class Monopod(omsf.DoubleIntegratorMotionPlatform):
    def __init__(self, **kwargs):
        n_axes = 1
        
        x, z, u = omsf.DoubleIntegratorMotionPlatform._makeVariables(n_axes)
        
        alpha = math.pi / 6
        self.r0 = numpy.array([math.cos(alpha), math.sin(alpha)])   # Coordinates of the point where the leg is attached to the ground, in 2D world frame
        self.r1 = numpy.array([0, 1])   # Coordinates of the point where the leg is attached to the platform, in 2D platform frame
        
        # Initial guess for finding P2W rotation
        self._guessR     = numpy.eye(2)
        self._guessRdot  = numpy.zeros((2, 2))
        self._guessRddot = numpy.zeros((2, 2))
        
        r0 = numpy.atleast_2d(self.r0).T
        r1 = numpy.atleast_2d(self.r1).T
        
        q_sym = cs.MX.sym('q')
        v_sym = cs.MX.sym('v')
        u_sym = cs.MX.sym('u')
        
        R_sym     = cs.MX.sym('R'    , 2, 2)
        Rdot_sym  = cs.MX.sym('Rdot' , 2, 2)
        Rddot_sym = cs.MX.sym('Rddot', 2, 2)
        
        # Function to calculate rotation matrix
        unitary_R = cs.mul(R_sym.T, R_sym) - cs.MX.eye(2)
        
        rhs = cs.vertcat([
            cs.quad_form(cs.mul(R_sym, r1) - r0) - q_sym ** 2,  # Leg length equation
            unitary_R[0, 0], unitary_R[0, 1], unitary_R[1, 1],  # R is a unitary matrix
        ])
        
        f = cs.MXFunction('Monopod_REquation', [cs.vec(R_sym), q_sym], [rhs])
        self._findR = cs.ImplicitFunction('Monopod_findR', 'newton', f)
        
        # Function to calculate derivative of rotation matrix
        unitary_Rdot  = cs.mul(Rdot_sym.T,  R_sym) + cs.mul(R_sym.T, Rdot_sym)
        
        rhs_dot = cs.vertcat([
                cs.mul(((cs.mul(R_sym, r1) - r0).T, Rdot_sym, r1)) - q_sym * v_sym, # Time-derivative of the leg length equation
                unitary_Rdot[0, 0], unitary_Rdot[0, 1], unitary_Rdot[1, 1] # Time-derivative of the unitarity condition
            ])
        
        f_dot = cs.MXFunction('Monopod_RdotEquation', [cs.vec(Rdot_sym), R_sym, q_sym, v_sym], [rhs_dot])
        self._findRdot = cs.ImplicitFunction('Monopod_findRdot', 'newton', f_dot)
        
        # Funciton to calculate 2nd derivative of rotation matrix
        unitary_Rddot = cs.mul(Rddot_sym.T, R_sym) + 2 * cs.mul(Rdot_sym.T, Rdot_sym) + cs.mul(R_sym.T, Rddot_sym)
        
        rhs_ddot = cs.vertcat([
                cs.mul((r1.T, Rdot_sym.T, Rdot_sym, r1)) + cs.mul(((cs.mul(R_sym, r1) - r0).T, Rddot_sym, r1)) - v_sym ** 2 - q_sym * u_sym,    # Second time-derivative of the leg length equation
                unitary_Rddot[0, 0], unitary_Rddot[0, 1], unitary_Rddot[1, 1]   # Second time-derivative of the unitarity condition
            ])
        
        f_ddot = cs.MXFunction('Monopod_RddotEquation', [cs.vec(Rddot_sym), Rdot_sym, R_sym, q_sym, v_sym, u_sym], [rhs_ddot])
        self._findRddot = cs.ImplicitFunction('Monopod_findRddot', 'newton', f_ddot)
        
        # Calculate symbolic rotation matrix
        [R_vec] = self._findR([cs.vec(self._guessR), x['q']])
        R = R_vec.reshape((2, 2))
        
        # Calculate symbolic derivative of rotation matrix
        [Rdot_vec] = self._findRdot([cs.vec(self._guessRdot), R, x['q'], x['v']])
        Rdot = Rdot_vec.reshape((2, 2))
        
        # Calculate symbolic 2nd derivative of transformation matrix
        [Rddot_vec] = self._findRddot([cs.vec(self._guessRddot), Rdot, R, x['q'], x['v'], u['u']])
        Rddot = Rddot_vec.reshape((2, 2))
        
        # Init outputFunction
        T = cs.MX.eye(4)            
        T[: 2, : 2] = R
        
        Tdot = cs.MX.zeros(4, 4)
        Tdot[: 2, : 2] = Rdot
        
        Tddot = cs.MX.zeros(4, 4)
        Tddot[: 2, : 2] = Rddot
        
        # Init the P2W function
        platform_to_world = cs.MXFunction('Monopod_P2W', [x, z, u], [T, Tdot, Tddot])
        
        # 
        omsf.DoubleIntegratorMotionPlatform.__init__(self, 
            motion_limits = [omsf.MotionLimits(0.1, 2, -1, 1, -1, 1)], 
            platform_to_world = platform_to_world, **kwargs)
    
class DoubleIntegratorMonopod(omsf.DoubleIntegratorMotionPlatform):
    def __init__(self, **kwargs):
        alpha = math.pi / 6
        self.r0 = numpy.array([math.cos(alpha), math.sin(alpha)])   # Coordinates of the point where the leg is attached to the ground, in 2D world frame
        self.r1 = numpy.array([0, 1])   # Coordinates of the point where the leg is attached to the platform, in 2D platform frame
        
        # Initial guess for finding P2W rotation
        self._guessR = numpy.eye(2)
        
        r0 = numpy.atleast_2d(self.r0).T
        r1 = numpy.atleast_2d(self.r1).T
        
        q_sym = cs.MX.sym('q')
        R_sym = cs.MX.sym('R', 2, 2)
        
        # Function to calculate rotation matrix
        unitary_R = cs.mul(R_sym.T, R_sym) - cs.MX.eye(2)
        
        rhs = cs.vertcat([
            cs.quad_form(cs.mul(R_sym, r1) - r0) - q_sym ** 2,  # Leg length equation
            unitary_R[0, 0], unitary_R[0, 1], unitary_R[1, 1],  # R is a unitary matrix
        ])
        
        f = cs.MXFunction('Monopod_REquation', [cs.vec(R_sym), q_sym], [rhs])
        self._findR = cs.ImplicitFunction('Monopod_findR', 'newton', f)
        
        # Calculate symbolic rotation matrix
        [R_vec] = self._findR([cs.vec(self._guessR), q_sym])
        R = R_vec.reshape((2, 2))
        
        # Init fT
        T = cs.MX.eye(4)            
        T[: 2, : 2] = R
        
        fT = cs.MXFunction('Monopod_fT', [q_sym], [T])
        
        omsf.DoubleIntegratorMotionPlatform.__init__(self, 
            motion_limits = [omsf.MotionLimits(0.1, 2, -1, 1, -1, 1)], fT = fT, **kwargs)