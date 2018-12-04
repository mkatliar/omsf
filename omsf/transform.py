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
Created on Oct 9, 2015

@author: mkatliar
'''
import casadi as cs
import numpy as np
import math


def mul(*args):
    return _mul(*args)


def inv(T):
    '''Inverse of the transform
    '''
    
    # Not using cs.inv(), so that no linear solver is involved => expand() works.
    invR = cs.transpose(getRotationMatrix(T))
    #return cs.vertcat(
    #    cs.horzcat(invR, cs.mtimes(invR, getTranslationVector(T))),
    #    cs.horzcat(cs.MX.zeros(1, 3), 1)
    #)
    return cs.vertcat(
        cs.horzcat(invR, -cs.mtimes(invR, getTranslationVector(T))),
        cs.horzcat(0, 0, 0, 1)
    )


def identity(dtype=float):
    '''Identity transform
    '''

    if _isSymbolic(dtype):
        return cs.MX.eye(4)
    else:
        return np.identity(4)

    
def translation(xyz):
    '''Translation by vector
    '''

    T = _eye(type(xyz[0]))        
    T[: 3, 3] = xyz                
    return T

    
def translationX(x):
    '''Translation along the X axis
    '''

    T = _eye(type(x))        
    T[0, 3] = x                
    return T

    
def translationY(y):
    '''Translation along the Y axis
    '''

    T = _eye(type(y))        
    T[1, 3] = y                
    return T

    
def translationZ(z):
    '''Translation along the Z axis
    '''

    T = _eye(type(z))        
    T[2, 3] = z                
    return T

    
def rotationZ(theta):
    '''Rotation around the Z axis
    '''

    T = _eye(type(theta))     
    T[0, 0] = T[1, 1] = _cos(theta)                
    T[1, 0] = _sin(theta)
    T[0, 1] = -T[1, 0]
    return T

    
def rotationY(theta):
    '''Rotation around the Y axis
    '''

    T = _eye(type(theta))     
    T[0, 0] = T[2, 2] = _cos(theta)                
    T[0, 2] = _sin(theta)
    T[2, 0] = -T[0, 2]
    return T
    
    
def rotationX(theta):
    '''Rotation around the X axis
    '''

    T = _eye(type(theta))     
    T[1, 1] = T[2, 2] = _cos(theta)                
    T[2, 1] = _sin(theta)
    T[1, 2] = -T[2, 1]
    return T

    
def denavitHartenberg(a, d, alpha, theta):
    '''Transform between two frames defined by Denavit-Hartenberg parameters
    '''

    return cs.mtimes([rotationZ(theta), translationZ(d), rotationX(alpha), translationX(a)])

    
def getRotationMatrix(T):
    '''Get rotation matrix form a 4x4 homogeneous transformation matrix.
    '''

    return T[: 3, : 3]


def getTranslationVector(T):
    '''Get rotation matrix form a 4x4 homogeneous transformation matrix.
    '''

    return T[: 3, 3]

    
_tolCos = 2. * math.cos(math.pi / 2.)
_tolSin = 2. * math.sin(math.pi)


def _eye(dtype):
    if _isSymbolic(dtype):
        return cs.MX.eye(4)
    else:
        return np.identity(4)

    
def _mul(*args):
    return cs.mtimes([*args])

    
def _cos(x):
    if _isSymbolic(type(x)):
        return cs.cos(x)
    else:
        c = math.cos(x)
        if math.fabs(c) <= _tolCos:
            return 0.
            
        return c

        
def _sin(x):
    if _isSymbolic(type(x)):
        return cs.sin(x)
    else:
        s = math.sin(x)
        if math.fabs(s) <= _tolSin:
            return 0.
            
        return s


def _isSymbolic(t):
    return t == cs.SX or t == cs.MX
