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
from .sensory_signal import SensorySignal
from omsf.plotting import _plot_vestibular_input


import numpy as np
import casadi as cs


# Given a time-dependent vector x and time vector t, computes dx/dt.
def time_derivative(x, t):
    # NOTE: There may be a smarter way to compute the derivative.
    n = x.shape[1]
    dx = np.zeros(x.shape)
    
    if n > 1:
        dx[:, 1]   = (x[:, 1  ] - x[:, 0  ]) / (t[1  ] - t[0  ])
        dx[:, n-1] = (x[:, n-1] - x[:, n-2]) / (t[n-1] - t[n-2])
    
    for j in range(1, n-1):
        dx[:, j] = (x[:, j+1] - x[:, j-1]) / (t[j+1] - t[j-1])

    return dx

# cumrect integrates y(x) using method of rectangles.
#   It is precise if y is a piecewise-constant function of x.
def cumrect(y, x):

    assert x.ndim == 1
    #assert(ismatrix(y));
    
    m = y.shape[0]
    dx = np.diff(x)
    
    return np.hstack([np.zeros([m, 1]), np.cumsum(y[:, : -1] * dx, axis = 1)])
    

def interpolate(t, t_base, x):
    assert x.shape[1] == len(t_base)
    
    d = x.shape[0]        
    u = np.zeros(d if np.isscalar(t) else [d, len(t)])
    
    for i in range(d):
        u[i] = np.interp(t, t_base, x[i, :])
        
    return u


def transformInertialSignal(y, T_BA):
    """Recalculate 9-dimensional inertial signal vector in a different coordinate frame.

    \param[y] 9-dimensional inertial signal vector. y = [f; omega; alpha]
    \param[T_BA] 4x4 matrix defining coordinate transformation FROM the new TO the original frame.
    """

    # TODO: use the INERTIAL_SIGNAL struct?

    f_A     = y[0 : 3]
    omega_A = y[3 : 6]
    alpha_A = y[6 : 9]
    
    # Recalculate from A to B
    R_BA = T_BA[: 3, : 3]
    r_BA = T_BA[: 3, 3]
    R_AB = cs.transpose(R_BA)
    
    f_B     = cs.mtimes(R_AB, f_A - cs.cross(alpha_A, r_BA) - cs.cross(omega_A, cs.cross(omega_A, r_BA)))
    omega_B = cs.mtimes(R_AB, omega_A)
    alpha_B = cs.mtimes(R_AB, alpha_A)
    
    return cs.vertcat(f_B, omega_B, alpha_B)


def inertialSignalFromTransformationMatrixAndDerivatives(T, dT, ddT):
    """Calculates linear acceleration, angular velocity and angular acceleraion
    from a 4x4 homogeneous transformation matrix
    and its 1st and 2nd derivatives.

    @param T 4x4 homogeneous transformation matrix from a moving frame (MF) to an inertial frame (IF)
    @param dT 1st time-derivative of T
    @param ddT 2nd time-derivative of T

    @return a, omega, alpha, where a is the acceleration, omega is the angular velocity
        and alpha is the angular acceleration, all expressed in the moving frame.
    """

    R = T[: 3, : 3]
    dR = dT[: 3, : 3]
    ddR = ddT[: 3, : 3]

    # Rotation matrix from IF to MF.
    R_T = cs.transpose(R)

    # Linear acceleration in MF.
    a = cs.mtimes(R_T, ddT[: 3, 3])

    # Rotational velocity in MF. 
    Omega = cs.mtimes(R_T, dR)
    omega = cs.vertcat(Omega[2, 1], Omega[0, 2], Omega[1, 0])
    
    # Rotational acceleration in MF.
    Alpha = cs.mtimes(cs.transpose(dR), dR) + cs.mtimes(R_T, ddR)
    alpha = cs.vertcat(Alpha[2, 1], Alpha[0, 2], Alpha[1, 0])

    return a, omega, alpha


def inertialSignalFromTransformationMatrix(T, q, dq, ddq):
    """Calculates linear acceleration, angular velocity and angular acceleraion
    from a 4x4 homogeneous transformation matrix depending on vector q,
    and 1st and 2nd derivatives of q.

    @param T 4x4 homogeneous transformation matrix from a moving frame (MF) to an inertial frame (IF)
    @param q a vector on which T depends
    @param dq 1st time-derivative of q
    @param ddq 2nd time-derivative of q

    @return a, omega, alpha, where a is the acceleration, omega is the angular velocity
        and alpha is the angular acceleration, all expressed in the moving frame.
    """

    # Time-derivative of the transformation matrix
    dT = cs.jtimes(T, q, dq)

    # Second time-derivative of the transformation matrix
    ddT = cs.jtimes(dT, q, dq) + cs.jtimes(T, q, ddq)

    return inertialSignalFromTransformationMatrixAndDerivatives(T, dT, ddT)


def recordedMotionToSensorySignal(motion, g_world_visual):
    '''Create sensory signal from IMU data.
    '''
    
    assert g_world_visual.shape == (3,)
    
    Nt = len(motion.time)
    g_dir = np.append(g_world_visual, 0)
    
    g_vis = np.zeros([4, Nt])
    for i in range(Nt):
        g_vis[:, i] = np.dot(motion.worldToLocal[:, :, i], g_dir)
    
    # TODO: rotational -> angular
    return SensorySignal(motion.time, v_v = motion.linearVelocity, omega_v = motion.rotationalVelocity, \
        g_v = g_vis[ : 3, :], f_i = -motion.gia, omega_i = motion.rotationalVelocity, alpha_i = motion.rotationalAcceleration)


def evaluateMotion(fig, sensory_signal, system_trajectory, param, scenario, step=0.01):
    """TODO: 'scenario' arg seems to be unused, remove?
    """
    _plot_vestibular_input(fig, sensory_signal.time, sensory_signal.inertialSignal(sensory_signal.time), 'r')

    t, ves_out = system_trajectory.inertialSignal.discretize(step)
    _plot_vestibular_input(fig, t, ves_out, 'b--')
  
    fig.gca().legend(['reference', 'actual'])

    
def calculateCost(cost_function, x1, t1, x2, t2):
    '''TODO: what do we do with this function?
    '''

    # Make a common time vector.
    t_begin = np.maximum(t1[ 0], t2[ 0])
    t_end   = np.minimum(t1[-1], t2[-1])
    t = np.union1d(t1, t2)
    t = np.union1d(t, [t_begin, t_end])
    t = t[np.logical_and(t >= t_begin, t <= t_end)]
    
    x1_new = interpolate(t, t1, x1)
    x2_new = interpolate(t, t2, x2)
    
    Nt = len(t)
    cost = np.zeros(Nt)
        
    for i in range(Nt):
        cost[i] = cost_function(x1_new[:, i], x2_new[:, i])
        
    return cost, t
    
