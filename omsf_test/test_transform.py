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
import numpy as np
import numpy.testing as nptest
import casadi as cs

from omsf import transform


class TransformTest(unittest.TestCase):
    """Unit tests for omsf.transform module
    """

    def test_inv(self):
        """Test omsf.transform.inv()
        """

        T_sym = cs.MX.sym('T', 4, 4)
        f = cs.Function('f', [T_sym], [cs.mtimes(T_sym, transform.inv(T_sym))])

        T = cs.mtimes([transform.rotationX(0.1), transform.rotationY(0.2), transform.rotationZ(0.3), transform.translation(np.array([0.4, 0.5, 0.6]))])

        nptest.assert_allclose(cs.evalf(f(T)), np.eye(4), atol=1e-16)


    def test_rotationY(self):
        """Test omsf.transform.rotationY()
        """
        T = transform.rotationY(2)

        nptest.assert_allclose(np.array(T), np.array([
            [-0.41614684,  0.,          0.90929743,  0.        ],
            [ 0.,          1.,          0.,          0.        ],
            [-0.90929743,  0.,         -0.41614684,  0.        ],
            [ 0.,          0.,          0.,          1.        ]]
        ), atol=1e-8)


    def test_rotationQ(self):
        """Test omsf.transform.rotationQ()
        """
        T = transform.rotationQ(np.array([0., 0., 0., 1.]))

        nptest.assert_allclose(np.array(T), np.array([
            [-1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1]]
        ), atol=1e-8)


    def test_getRotationAngle(self):
        """Test omsf.transform.getRotationAngle()
        """
        T = transform.rotationQ([0.9689124, 0.0661215, 0.132243, 0.1983645])
        self.assertAlmostEqual(transform.getRotationAngle(T), 0.5, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
