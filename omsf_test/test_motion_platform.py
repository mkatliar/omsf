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

import omsf
import numpy as np
import casadi as cs
import casadi_extras as ct
from motion_platform_x import MotionPlatformX

DATA_DIR = 'omsf_test/data'


class Test(unittest.TestCase):
    def test_headFrameGravity(self):
        
        platform = MotionPlatformX()
        
        x = ct.struct_MX(platform.state.expr)
        x['q'] = cs.SX.sym('q')
        x['v'] = cs.DM.zeros(platform.numberOfAxes)

        g_head = platform.headFrameGravity(x=x.cat, g=omsf.DEFAULT_GRAVITY, T_HP=omsf.transform.identity())['g']
        
        # Check that the gravity is pointing strictly downwards.
        self.assertEqual(g_head[0], 0)
        self.assertEqual(g_head[1], 0)
        self.assertLess(g_head[2], 0)

    
if __name__ == "__main__":
    unittest.main()