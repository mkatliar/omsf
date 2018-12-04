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
import unittest
import os
import numpy as np, numpy.testing

import omsf.util as util
import omsf.transform as tr
from omsf.math import crossProductMatrix
import casadi as cs

DATA_DIR = 'omsf_test/data'


class Test(unittest.TestCase):


    def test_cumrect(self):
        k = 0.1
        x = k * np.arange(1, 4)
        y = np.vstack([range(1, 4), [2, 4, 6]])
        
        z = util.cumrect(y, x)
        
        np.testing.assert_allclose(z, np.array([[0, 0], [1, 2], [3, 6]]).T * k)


class TestTransformInertialSignal(unittest.TestCase):

    def test_specificForceRotation(self):
        """Test rotation of specific force when angular velocity and acceleration are 0"""

        T_ab = tr.rotationZ(np.pi / 2)
        y_a = cs.DM([1, 2, 3, 0, 0, 0, 0, 0, 0])
        y_b = util.transformInertialSignal(y_a, T_ab)

        np.testing.assert_array_almost_equal(y_b, cs.DM([2, -1, 3, 0, 0, 0, 0, 0, 0]))


    def test_centrifugal(self):
        """Test centrifugal force"""

        y_a = cs.DM([0, 0, 0, 0, 0, 2, 0, 0, 0])
        T_ab = tr.translation([3, 0, 0])
        y_b = util.transformInertialSignal(y_a, T_ab)
            
        np.testing.assert_array_almost_equal(y_b, cs.DM([12, 0, 0, 0, 0, 2, 0, 0, 0]))


    def test_euler(self):
        """Test force due to Euler acceleration"""

        y_a = cs.DM([0, 0, 0, 0, 0, 0, 0, 0, 3])
        T_ab = tr.translation([4, 0, 0])
        y_b = util.transformInertialSignal(y_a, T_ab)

        np.testing.assert_array_almost_equal(y_b, cs.DM([0, -12, 0, 0, 0, 0, 0, 0, 3]))


    def test_combined(self):
        """Test a combination of coordinate frame rotation, centrifugal and Euler force"""
        T_ab = tr.mul(tr.translation([3, 0, 0]), tr.rotationZ(np.pi / 2))
        y_a = cs.DM([1, 2, 3, 0, 0, 2, 0, 0, 3])
        y_b = util.transformInertialSignal(y_a, T_ab)

        np.testing.assert_array_almost_equal(y_b, cs.DM([2 + (-9), -1 + (-12), 3, 0, 0, 2, 0, 0, 3]))

        
class TestInertialSignalFromTransformationMatrixAndDerivatives(unittest.TestCase):

    def test_againstTrustedImplementation(self):
        """Test that the new implementation with simplified formulas 
        gives the same result as the one that we trust.
        """

        '''
        def makeTransform(R, r):
            return cs.vertcat(cs.horzcat(R, r), cs.horzcat([0, 0, 0, 1]))        
        '''

        # 6-dimensional position-attitude vector
        q = cs.MX.sym('q', 6)

        # Transformation matrix
        T = cs.mtimes([tr.translation(q[: 3]), tr.rotationZ(q[5]), tr.rotationY(q[4]), tr.rotationX(q[3])])

        # Derivative of the transformation matrix
        dq = cs.MX.sym('dq', 6)
        dT = cs.jtimes(T, q, dq)

        # Second derivative of the transformation matrix
        ddq = cs.MX.sym('ddq', 6)
        ddT = cs.jtimes(dT, q, dq) + cs.jtimes(dT, dq, ddq)

        # Evaluate matrices
        f = cs.Function('f', [q, dq, ddq], [T, dT, ddT])
        q_val = np.random.rand(6)
        dq_val = np.random.rand(6)
        ddq_val = np.random.rand(6)
        [T_val, dT_val, ddT_val] = f(q_val, dq_val, ddq_val)

        # Calculate inertial signal components using the normal implementation.
        a, omega, alpha = util.inertialSignalFromTransformationMatrixAndDerivatives(T_val, dT_val, ddT_val)

        # Calculate inertial signal components using the reference implementation.
        a_ref, omega_ref, alpha_ref = self._trustedImplementation(T_val, dT_val, ddT_val)

        # Test that the values are the same.
        np.testing.assert_allclose(a, a_ref)
        np.testing.assert_allclose(omega, omega_ref)
        np.testing.assert_allclose(alpha, alpha_ref)
        

    def _trustedImplementation(self, T, T1, T2):
        """Calculates linear acceleration, angular velocity and angular acceleraion
        from a 4x4 homogeneous transformation matrix 
        and its 1st and 2nd derivatives.

        @param T 4x4 homogeneous transformation matrix from a moving frame to an inertial frame
        @param T1 1st time-derivative of T
        @param T2 2nd time-derivative of T

        @return a, omega, alpha, where a is the acceleration, omega is the angular velocity
            and alpha is the angular acceleration, all expressed in the moving frame.
        """

        R1 = T1[: 3, : 3]
        R2 = T2[: 3, : 3]

        # Rotation matrix from WF to PF.
        R_WP = cs.transpose(T[: 3, : 3])

        # Linear acceleration in WF.
        a_W = T2[: 3, 3]

        # Linear acceleration in PF.
        a_P = cs.mtimes(R_WP, a_W)

        # Rotational velocity in WF.
        Omega_W = cs.mtimes(R1, R_WP)
        #omega_W = [Omega_W(3, 2) - Omega_W(2, 3); Omega_W(1, 3) - Omega_W(3, 1); Omega_W(2, 1) - Omega_W(1, 2)] / 2;
        omega_W = cs.vertcat(Omega_W[2, 1], Omega_W[0, 2], Omega_W[1, 0])
        
        # Rotational velocity in PF. 
        omega_P = cs.mtimes(R_WP, omega_W)
        
        # Rotational acceleration in WF.
        Omega_dot_W = cs.mtimes(R2, R_WP) - cs.mtimes(Omega_W, Omega_W)
        omega_dot_W = cs.vertcat(Omega_dot_W[2, 1], Omega_dot_W[0, 2], Omega_dot_W[1, 0])
        
        # Rotational acceleration in PF.
        omega_dot_P = cs.mtimes(R_WP, omega_dot_W)

        return a_P, omega_P, omega_dot_P


class TestInertialSignalFromTransformationMatrix(unittest.TestCase):

    def test(self):
        """Test that the new implementation with simplified formulas 
        gives the same result as the one that we trust.
        """

        # 6-dimensional position-attitude vector
        q = cs.MX.sym('q', 6)

        # Transformation matrix
        T = cs.mtimes([tr.translation(q[: 3]), tr.rotationZ(q[5]), tr.rotationY(q[4]), tr.rotationX(q[3])])

        # Derivative of the transformation matrix
        dq = cs.MX.sym('dq', 6)
        dT = cs.jtimes(T, q, dq)

        # Second derivative of the transformation matrix
        ddq = cs.MX.sym('ddq', 6)
        ddT = cs.jtimes(dT, q, dq) + cs.jtimes(dT, dq, ddq)

        # Evaluate matrices
        f = cs.Function('f', [q, dq, ddq], [T, dT, ddT])
        q_val = np.random.rand(6)
        dq_val = np.random.rand(6)
        ddq_val = np.random.rand(6)
        [T_val, dT_val, ddT_val] = f(q_val, dq_val, ddq_val)

        # Calculate inertial signal components using inertialSignalFromTransformationMatrixAndDerivatives().
        a_ref, omega_ref, alpha_ref = util.inertialSignalFromTransformationMatrixAndDerivatives(T_val, dT_val, ddT_val)

        # Calculate inertial signal components using inertialSignalFromTransformationMatrix().
        a, omega, alpha = util.inertialSignalFromTransformationMatrix(T, q, dq, ddq)
        f1 = cs.Function('f1', [q, dq, ddq], [a, omega, alpha])
        a_val, omega_val, alpha_val = f1(q_val, dq_val, ddq_val)

        # Test that the values are the same.
        np.testing.assert_allclose(a_val, a_ref)
        np.testing.assert_allclose(omega_val, omega_ref)
        np.testing.assert_allclose(alpha_val, alpha_ref)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_cumrect']
    unittest.main()