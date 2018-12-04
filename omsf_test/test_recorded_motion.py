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
Created on Oct 21, 2016

@author: kotlyar
'''
import unittest
import numpy as np
import numpy.matlib as matlib
import scipy.signal
import matplotlib.pyplot as plt
import os

import omsf.io.viasync
import omsf.plotting

DATA_DIR = 'omsf_test/data'

class TestRecordedMotion(unittest.TestCase):


    def testCentrifugalAcceleration(self):
        v           = np.atleast_2d([0, 0, 0]).T
        omega       = np.atleast_2d([0, 0, 2]).T
        gia         = np.atleast_2d([0, 0, 0]).T
        omega_dot   = np.atleast_2d([0, 0, 0]).T
        world_to_local = np.atleast_3d(np.identity(4))
        t           = np.array([0])
        
        mot = omsf.RecordedMotion(t, v = v, omega = omega, omega_dot = omega_dot, gia = gia, world_to_local = world_to_local)        
        a_c = mot.CentrifugalAcceleration(np.array([3, 0, 0]))
        
        np.testing.assert_equal(a_c, np.atleast_2d([-12, 0, 0]).T)
        
    def testEulerAcceleration(self):
        v           = np.atleast_2d([0, 0, 0]).T
        omega       = np.atleast_2d([0, 0, 2]).T
        gia         = np.atleast_2d([0, 0, 0]).T
        omega_dot   = np.atleast_2d([0, 0, 3]).T
        world_to_local = np.atleast_3d(np.identity(4))
        t           = np.array([0])
        
        mot = omsf.RecordedMotion(t, v = v, omega = omega, omega_dot = omega_dot, gia = gia, world_to_local = world_to_local)
        a_eul = mot.EulerAcceleration(np.array([4, 0, 0]))
        
        np.testing.assert_equal(a_eul, np.atleast_2d([0, 12, 0]).T)
        
    def testChangeReferenceFrame(self):
        # Load test IMU data.
        motion = omsf.io.viasync.loadRecordedMotion(os.path.join(DATA_DIR, "Brake.mat"))
        
        # Create a non-trivial transform matrix.
        T = omsf.transform.mul(omsf.transform.translation(np.array([0.15, 0.20, 0.30])), 
            omsf.transform.rotationZ(0.1), omsf.transform.rotationY(0.2), omsf.transform.rotationX(0.1))
        
        # transform data from frame 0 to frame 1.
        motion1 = motion.ChangeReferenceFrame(T)
        
        # Verify that the result is different.
        self.assertTrue(np.any(np.not_equal(motion1.gia, motion.gia)))
        self.assertTrue(np.any(np.not_equal(motion1.rotationalVelocity, motion.rotationalVelocity)))
        self.assertTrue(np.any(np.not_equal(motion1.rotationalAcceleration, motion.rotationalAcceleration)))
        np.testing.assert_array_equal(motion1.time, motion.time)
        
        # transform data back from frame 1 to frame 0.
        motion2 = motion1.ChangeReferenceFrame(omsf.transform.inv(T))
        
        # Verify that the result of 2-way transform is the same as original.
        np.testing.assert_array_almost_equal(motion2.gia, motion.gia)
        np.testing.assert_array_almost_equal(motion2.rotationalVelocity, motion.rotationalVelocity)
        np.testing.assert_array_almost_equal(motion2.rotationalAcceleration, motion.rotationalAcceleration)
        np.testing.assert_array_equal(motion2.time, motion.time)
        
    def testSampleRateOneSample(self):
        v           = np.atleast_2d([0, 0, 0]).T
        omega       = np.atleast_2d([0, 0, 2]).T
        gia         = np.atleast_2d([0, 0, 0]).T
        omega_dot   = np.atleast_2d([0, 0, 0]).T
        world_to_local = np.atleast_3d(np.identity(4))
        t           = np.array([0])
        
        mot = omsf.RecordedMotion(t, v = v, omega = omega, omega_dot = omega_dot, gia = gia, world_to_local = world_to_local)
        
        self.assertEqual(mot.sampleRate, None)
        
    def testSampleRateUniform(self):
        v           = np.atleast_2d([0, 0, 0]).T
        omega       = np.atleast_2d([0, 0, 2]).T
        gia         = np.atleast_2d([0, 0, 0]).T
        omega_dot   = np.atleast_2d([0, 0, 0]).T
        world_to_local = np.atleast_3d(np.identity(4))
        t           = np.array([0, 0.1, 0.2])
        N = len(t)
        
        mot = omsf.RecordedMotion(t, v = matlib.repmat(v, 1, N), 
                                  omega = matlib.repmat(omega, 1, N), 
                                  omega_dot = matlib.repmat(omega_dot, 1, N), 
                                  gia = matlib.repmat(gia, 1, N), 
                                  world_to_local = np.tile(world_to_local, [1, 1, N]))
        
        self.assertEqual(mot.sampleRate, 10)
        
    def testSampleRateNonUniform(self):
        v           = np.atleast_2d([0, 0, 0]).T
        omega       = np.atleast_2d([0, 0, 2]).T
        gia         = np.atleast_2d([0, 0, 0]).T
        omega_dot   = np.atleast_2d([0, 0, 0]).T
        world_to_local = np.atleast_3d(np.identity(4))
        t           = np.array([0, 0.15, 0.2])
        N = len(t)
        
        mot = omsf.RecordedMotion(t, v = matlib.repmat(v, 1, N), 
                                  omega = matlib.repmat(omega, 1, N), 
                                  omega_dot = matlib.repmat(omega_dot, 1, N), 
                                  gia = matlib.repmat(gia, 1, N), 
                                  world_to_local = np.tile(world_to_local, [1, 1, N]))
        
        self.assertEqual(mot.sampleRate, None)
        
    def testFilter(self):
        # Load test IMU data.
        motion = omsf.io.viasync.loadRecordedMotion(os.path.join(DATA_DIR, "Brake.mat"))
        
        f1 = plt.figure()
        omsf.plotting.plotRecordedMotion(motion, f1)
        f1.suptitle('Original motion')
        
        # Design the low-pass filter
        fs = motion.sampleRate
        f_pass = 3.
        f_stop = 10.
        b, a = scipy.signal.iirdesign(wp = 2 * f_pass / fs, ws = 2 * f_stop / fs, gpass = 1, gstop = 20, analog=False, ftype='butter', output='ba')
        
        # Filter the data.
        motion1 = motion.Filter(b = b, a = a, what = 'all', cut = True)
        
        # Plot the result.
        f2 = plt.figure()
        omsf.plotting.plotRecordedMotion(motion1, f2)
        f2.suptitle('Filtered motion')
        plt.show()
        
    def testSelect(self):
        v           = np.atleast_2d([0, 0, 0]).T
        omega       = np.atleast_2d([0, 0, 2]).T
        gia         = np.atleast_2d([0, 0, 0]).T
        omega_dot   = np.atleast_2d([0, 0, 0]).T
        world_to_local = np.atleast_3d(np.identity(4))
        t           = np.array([0, 0.15, 0.2])
        N = len(t)
        
        mot = omsf.RecordedMotion(t, v = matlib.repmat(v, 1, N), 
                                  omega = matlib.repmat(omega, 1, N), 
                                  omega_dot = matlib.repmat(omega_dot, 1, N), 
                                  gia = matlib.repmat(gia, 1, N), 
                                  world_to_local = np.tile(world_to_local, [1, 1, N]))
        
        ind = [1, 2]
        mot1 = mot.Select(ind)
        
        np.testing.assert_array_equal(mot1.time, mot.time[ind])
        np.testing.assert_array_equal(mot1.gia, mot.gia[:, ind])
        np.testing.assert_array_equal(mot1.linearVelocity, mot.linearVelocity[:, ind])
        np.testing.assert_array_equal(mot1.rotationalVelocity, mot.rotationalVelocity[:, ind])
        np.testing.assert_array_equal(mot1.rotationalAcceleration, mot.rotationalAcceleration[:, ind])
        np.testing.assert_array_equal(mot1.worldToLocal, mot.worldToLocal[:, :, ind])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()