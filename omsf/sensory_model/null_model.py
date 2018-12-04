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


class NullModel(object):
    '''Sensory model with empty state and no output.
    '''

    def __init__(self):
        '''Constructor
        '''

        self.state = ct.BoundedVariable()
        self.stateDerivative = cs.MX.sym('xdot', 0)
        self.algState = ct.BoundedVariable()
        self.dae = cs.MX.sym('dae', 0)
