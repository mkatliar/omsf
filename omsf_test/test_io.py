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
Created on Jan 26, 2015

@author: mkatliar
'''
import unittest
import numpy.testing, tempfile, os
import matplotlib.pyplot as plt

import omsf.io.matlab
import omsf.io.viasync
import omsf.io.carsim
import omsf.plotting

DATA_DIR = 'omsf_test/data'

class TestMatlab(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(suffix = '.mat') as f:
            self.fileName = f.name

    def test_LoadSaveRecordedMotion(self):
        v           = numpy.atleast_2d([1, 2, 3]).T
        omega       = numpy.atleast_2d([4, 5, 6]).T
        gia         = numpy.atleast_2d([0, 0, -9.8]).T
        omega_dot   = numpy.atleast_2d([0, 0, 0]).T
        world_to_local = numpy.atleast_3d(numpy.identity(4))
        t           = numpy.array([0])
        
        mot = omsf.RecordedMotion(t, v = v, omega = omega, omega_dot = omega_dot, gia = gia, world_to_local = world_to_local)

        omsf.io.matlab.SaveRecordedMotion(mot, self.fileName)
        mot1 = omsf.io.matlab.loadRecordedMotion(self.fileName)
            
        numpy.testing.assert_equal(mot1.linearVelocity, mot.linearVelocity)
        numpy.testing.assert_equal(mot1.gia, mot.gia)
        numpy.testing.assert_equal(mot1.rotationalAcceleration, mot.rotationalAcceleration)
        numpy.testing.assert_equal(mot1.rotationalVelocity, mot.rotationalVelocity)
        numpy.testing.assert_equal(mot1.time, mot.time)
        
    def test_LoadSaveSensoryInput(self):
        v0 = numpy.array([[0, 0, 0], [1, 2, 3]]).T
        f = numpy.array([[1, 0, 0], [4, 5, 6]]).T
        omega = numpy.array([[0, 0, 0], [7, 8, 9]]).T
        g = numpy.array([[0, 0, -1], [-1, -1, 2]]).T
        t = numpy.array([0, 1])
        
        si = omsf.SensorySignal(t, v_v = v0, omega_v = omega, g_v = g, f_i = f, omega_i = omega)

        omsf.io.matlab.SaveSensoryInput(si, self.fileName)
        si1 = omsf.io.matlab.LoadSensoryInput(self.fileName)
            
        numpy.testing.assert_equal(si1.v_v, si.v_v)
        numpy.testing.assert_equal(si1.omega_v, si.omega_v)
        numpy.testing.assert_equal(si1.g_v, si.g_v)
        numpy.testing.assert_equal(si1.f_i, si.f_i)
        numpy.testing.assert_equal(si1.omega_i, si.omega_i)
        numpy.testing.assert_equal(si1.alpha_i, si.alpha_i)
        numpy.testing.assert_equal(si1.time, si.time)
        
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        os.remove(self.fileName)
        
class TestViaSync(unittest.TestCase):

    def test_LoadRecordedMotion(self):
        motion = omsf.io.viasync.loadRecordedMotion(os.path.join(DATA_DIR, "Brake.mat"))
        omsf.plotting.plotRecordedMotion(motion, plt.figure())
        plt.show()
        
class TestCarSim(unittest.TestCase):

    def test_LoadRecordedMotion(self):
        motion = omsf.io.carsim.loadRecordedMotion(os.path.join(DATA_DIR, "TestCarSim.txt"))
        omsf.plotting.plotRecordedMotion(motion, plt.figure())
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_LoadSaveMat']
    unittest.main()