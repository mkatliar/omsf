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
import numpy as np, numpy.testing
import casadi as cs

import omsf.math

class Test(unittest.TestCase):


    def test_quatR(self):
        q = np.array([1, 0, 0, 0]);        
        R = np.eye(3, 3)
        
        numpy.testing.assert_allclose(omsf.math.quatR(q), R)


    def test_vectorInterpolant(self):
        '''Check if vectorInterpolant() function works.
        '''

        M = 10
        N = 3
        x = np.arange(M)
        val = [np.random.rand(M) for _ in range(N)]
        method = 'bspline'

        si = [cs.interpolant('si_{0}'.format(i), method, [x], v) for i, v in enumerate(val)]

        vi = [
            omsf.math.vectorInterpolant('vi', method, [x], val),    # Check with nested list argument
            omsf.math.vectorInterpolant('vi', method, [x], np.array(val)),  # Check with numpy.array argument
            #omsf.math.vectorInterpolant('vi', method, [x], cs.DM(val))  # Check with casadi.DM argument
        ]

        for xq in np.linspace(min(x), max(x), 10 * M):
            for vii in vi:
                numpy.testing.assert_equal(vii(xq), np.atleast_2d([sii(xq) for sii in si]).T)


if __name__ == "__main__":
    unittest.main()