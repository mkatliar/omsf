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
import casadi_extras as ct
import casadi as cs

import numpy as np, scipy.signal


class System:
    '''
    Generic system class.
    '''

    def __init__(self, state_derivative, input, state, alg_state, dae, output):
        self.input = input
        self.state = state
        self.algState = alg_state
        self.output = output
        self.dae = dae
        self.stateDerivative = state_derivative
        #self.daeFunction = dae_function
        #self.outputFunction = output_function
        self.Nu = input.numel()
        self.Nx = state.numel()
        self.Ny = output.numel()

        
    def FindSteadyState(self, u):
        '''
        Finds the steady state (x, z) for input u, such that f_ode(x, z, u) = 0, f_alg(x, z, u) = 0
        '''
        raise NotImplementedError('FindSteadyState() for generic System is not implemented yet')


class LTISystem(System):
    '''
    classdocs
    '''

    def __init__(self, A, B, C, D, input_struct=None, state_struct=None, output_struct=None):
        '''
        Constructor
        '''
        
        # Checking the arguments
        Nx = A.shape[1]
        assert A.shape == (Nx, Nx)
        
        Nu = B.shape[1]
        assert B.shape == (Nx, Nu)
        
        Ny = C.shape[0]
        assert C.shape == (Ny, Nx)
        
        assert D.shape == (Ny, Nu) 
        
        if input_struct is not None:
            assert input_struct.size == Nu
            input = input_struct
        else:
            input = cs.MX.sym('u', Nu)
        input = ct.Inequality(expr=input, nominal=cs.DM.zeros(Nu))
            
        if state_struct is not None:
            assert state_struct.size == Nx
            state = state_struct
        else:
            state = cs.MX.sym('x', Nx)
        state = ct.Inequality(expr=state, nominal=cs.DM.zeros(Nx))

        xdot = cs.MX.sym('xdot', state.numel())
        dae = xdot - (cs.mtimes(A, state.expr) + cs.mtimes(B, input.expr))
        
        output = cs.mtimes(C, state.expr) + cs.mtimes(D, input.expr)
        if output_struct is not None:
            output = output_struct(output)
        
        # Initializing base class
        System.__init__(self, state_derivative=xdot, input=input, state=state, alg_state=ct.BoundedVariable(), 
            dae=dae, output=output)
        
        # Remembering A, B, C, D
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    
    def getTf(self, *args):
        if args:
            i = args[0]
            j = args[1]
            return scipy.signal.lti(self.A, np.atleast_2d(self.B[:, j]), np.atleast_2d(self.C[i, :]), np.atleast_2d(self.D[i, j]))
        else:            
            tf = []
            for ii in range(self.A.shape[0]):
                tf.append([])
                for jj in range(self.B.shape[1]):
                    tf[-1].append(self.getTf(ii, jj))
                
        return tf

    
    def FindSteadyState(self, u):
        '''
        Ax + Bu = 0 => x = A \ (-Bu)
        '''
        return np.linalg.solve(self.A, -cs.mtimes(self.B, u))