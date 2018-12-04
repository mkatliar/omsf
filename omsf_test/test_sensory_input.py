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

import omsf

import unittest
import numpy.testing
import casadi as cs

        
class TestSensoryInput(unittest.TestCase):
    def test_init(self):
        v0 = numpy.array([[0, 0, 0], [1, 2, 3]]).T
        f = numpy.array([[1, 0, 0], [4, 5, 6]]).T
        omega = numpy.array([[0, 0, 0], [7, 8, 9]]).T
        g = numpy.array([[0, 0, -1], [-1, -1, 2]]).T
        t = numpy.array([0, 1])
        
        si = omsf.SensorySignal(t, v_v = v0, omega_v = omega, g_v = g, f_i = f, omega_i = omega)
        
        self.assertEqual(si.startTime, t[0])
        self.assertEqual(si.stopTime, t[-1])
        numpy.testing.assert_array_equal(si.visualLinearVelocity(t), v0)
        numpy.testing.assert_array_equal(si.visualAngularVelocity(t), omega)
        numpy.testing.assert_array_equal(si.visualDirectionOfGravity(t), g)
        numpy.testing.assert_array_equal(si.inertialSpecificForce(t), f)
        numpy.testing.assert_array_equal(si.inertialAngularVelocity(t), omega)


if __name__ == "__main__":
    unittest.main()