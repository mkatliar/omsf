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
import numpy as np
import scipy.signal
import omsf.util as util

'''
RecordedMotion represents inertial data in some frame of reference.
The inertial data consists of 
- linear velocity v
- gravito-inertial acceleration gia
- rotational velocity omega
- rotational acceleration omega_dot
- 4x4 transformation matrix from the world frame to the frame where the other quantities are represented world_to_local
- time vector t
'''
class RecordedMotion:
    # RecordedMotion(t, v, gia, omega, omega_dot, world_to_local)
    def __init__(self, t, **kwargs):
        assert t.ndim == 1
        Nt = len(t)
        
        if 'v' in kwargs:
            v = kwargs['v']
            assert v.shape == (3, Nt)
            self.linearVelocity = v
        else:
            self.linearVelocity = None
        
        gia = kwargs['gia']
        assert gia.shape == (3, Nt)
            
        omega = kwargs['omega']
        assert omega.shape == (3, Nt)
        
        if 'world_to_local' in kwargs:
            world_to_local = kwargs['world_to_local']
            assert world_to_local.shape == (4, 4, Nt)
            self.worldToLocal = world_to_local
        else:
            self.worldToLocal = None
        
        if 'omega_dot' in kwargs:
            omega_dot = kwargs['omega_dot']
            assert omega_dot.shape == (3, Nt)
        else:
            omega_dot = util.time_derivative(omega, t)              
        
        self.gia = gia
        self.rotationalVelocity = omega
        self.rotationalAcceleration = omega_dot
        self.time = t
        
        # Determine sample rate from the time vector.
        self.sampleRate = None
        if Nt > 1:
            dt = np.diff(t)
            if np.amax(np.fabs(dt - np.mean(dt))) < 1e-10:
                dt = (t[-1] - t[0]) / (Nt - 1)
                self.sampleRate = 1 / dt
        
    def Cut(self, t1, t2):
        ind = np.logical_and(self.time >= t1, self.time <= t2)

        if self.linearVelocity is not None:
            v = self.linearVelocity[:, ind]
        else:
            v = None
        
        if self.worldToLocal is not None:
            world_to_local = self.worldToLocal[:, :, ind]
        else:
            world_to_local = None
        
        return RecordedMotion(self.time[ind], v = v, gia = self.gia[:, ind], omega = self.rotationalVelocity[:, ind], \
                                      omega_dot = self.rotationalAcceleration[:, ind], world_to_local = world_to_local)
                                      
    def length(self):
        return len(self.time)
        
    """ 
    \brief Compute centrifugal acceleration in point defined by vector r in IMU coordinate frame.
    
    Output is in IMU coordinate frame.
    """                              
    def CentrifugalAcceleration(self, r):
        return np.cross(self.rotationalVelocity, np.cross(self.rotationalVelocity, r, axisa = 0), axisa = 0).T
    
    """
    \brief Compute Euler acceleration in point defined by vector r in IMU coordinate frame.

    Output is in IMU coordinate frame.
    """
    def EulerAcceleration(self, r):
        return np.cross(self.rotationalAcceleration, r, axisa = 0).T
    
    """
    \brief Convert IMU measurement to other frame of reference.
        
    \param T_imu_target is a 4x4 static transformation matrix from IMU frame to a target frame.
    \returns IMU data in target frame.
    """
    def ChangeReferenceFrame(self, T_imu_target):
        # Transform matrix from target frame to IMU frame.
        T_target_imu = np.linalg.inv(T_imu_target)
    
        # Rotation matrix from IMU frame to target frame.
        R_imu_target = T_imu_target[: 3, : 3]
    
        # Origin of the target frame in IMU frame, repeated across time.
        n = self.length()
        r = T_target_imu[: 3, 3]
        
        # Compute linear velocity in target frame.
        v_out = np.dot(R_imu_target, self.linearVelocity)
    
        # Compute linear acceleration in target frame.
        gia_out = np.dot(R_imu_target, self.gia + self.EulerAcceleration(r) + self.CentrifugalAcceleration(r))
        
        # Compute rotational velocity in target frame.
        omega_out = np.dot(R_imu_target, self.rotationalVelocity)
        
        # Compute rotational acceleration in target frame.
        omega_dot_out = np.dot(R_imu_target, self.rotationalAcceleration)
        
        world_to_local = np.zeros(self.worldToLocal.shape)
        for i in range(n):
            world_to_local[:, :, i] = np.dot(T_imu_target, self.worldToLocal[:, :, i])
    
        return RecordedMotion(t = self.time, v = v_out, gia = gia_out, omega = omega_out, omega_dot = omega_dot_out, world_to_local = world_to_local)

    """
    \brief Filter recorded motion using filter b/a.
    
    \param[b] Filter transfer function is b/a
    \param[a] Filter transfer function is b/a
    \param[what] -- specifies which components to filter. Can be 'all', 'gia' or 'omega'.
    \param[cut] If cut == true, RecordedMotion will be cut at the beginning and at
    the end to remove transients, according to filter's impulse response
    length.
    """
    def Filter(self, b, a, what = 'all', cut = False):
        if what == 'all' or what == 'gia':
            gia = scipy.signal.filtfilt(b, a, self.gia, axis = 1)
        else:
            gia = self.gia
        
        if what == 'all' or what == 'omega':
            omega = scipy.signal.filtfilt(b, a, self.rotationalVelocity, axis = 1)
            alpha = scipy.signal.filtfilt(b, a, self.rotationalAcceleration, axis = 1)
        else:
            omega = self.rotationalVelocity
            alpha = self.rotationalAcceleration
            
        result = RecordedMotion(t = self.time, gia = gia, omega = omega, omega_dot = alpha, world_to_local = self.worldToLocal, v = self.linearVelocity)
        
        if cut:
            tout, _ = scipy.signal.dimpulse((b, a, 1. / self.sampleRate))
            n_cut = len(tout)
            i = np.arange(self.length())
            result = result.Select(np.logical_and(i + 1 > n_cut, i <= i[-1] - n_cut))
            
        return result
    
    """
    \brief Select samples defined by index ind.
    """
    def Select(self, ind):
        if self.linearVelocity is not None:     
            v = self.linearVelocity[:, ind]
        else:
            v = None
            
        if self.worldToLocal is not None:                
            world_to_local = self.worldToLocal[:, :, ind]
        else:
            world_to_local = None
        
        return RecordedMotion(t = self.time[ind], v = v, gia = self.gia[:, ind], omega = self.rotationalVelocity[:, ind],
            omega_dot = self.rotationalAcceleration[:, ind], world_to_local = world_to_local)
                                              