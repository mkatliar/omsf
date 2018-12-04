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
import casadi_extras as ct


'''Defines component order of the inertial signal.
'''
INERTIAL_SIGNAL = ct.struct_symMX([
    ct.entry('f', shape=3),
    ct.entry('omega', shape=3),
    ct.entry('alpha', shape=3)
])


'''Symbolic variable for the reference inertial signal.
'''
REFERENCE_INERTIAL_SIGNAL = ct.struct_symMX(INERTIAL_SIGNAL)


'''Symbolic variable for inertial signal error.
'''
INERTIAL_SIGNAL_ERROR = INERTIAL_SIGNAL(INERTIAL_SIGNAL.cat - REFERENCE_INERTIAL_SIGNAL.cat)


'''Defines component order of the visual signal.
'''
VISUAL_SIGNAL = ct.struct_symMX([
    ct.entry('v', shape = 3),
    ct.entry('omega', shape = 3),
    ct.entry('g', shape = 3)
])
