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
from omsf.sensory_model.ormsby_model import OrmsbyModel

import unittest
import numpy.testing
import casadi as cs


class TestOrmsbyModel(unittest.TestCase):

    def test_init(self):
        o = OrmsbyModel()
        
        t_oto = numpy.arange(0, 30, 1)
        impulse_oto = [-0.917431192660551, -0.751129131264204, -0.614972519298752, -0.503496913847731, -0.412228407446993, -0.337504074469213, -0.276324965057066, -0.226235746735419, -0.185226163297849, -0.151650356166593, -0.124160810308819, -0.101654273726912, -0.0832274800820299, -0.0681408974443431, -0.0557890482800166, -0.0456762095118019, -0.0373965174113452, -0.0306176788626845, -0.0250676352727455, -0.0205236439047391, -0.0168033384300314, -0.0137574099270438, -0.0112636145899711, -0.00922186765562716, -0.00755022665047711, -0.00618160275145457, -0.00506106827592733, -0.00414365224092906, -0.00339253551970912, -0.00277757316089525]
        
        numpy.testing.assert_almost_equal(o.getTf(0, 0).impulse(T = t_oto), (t_oto, impulse_oto))
        numpy.testing.assert_almost_equal(o.getTf(1, 1).impulse(T = t_oto), (t_oto, impulse_oto))
        numpy.testing.assert_almost_equal(o.getTf(2, 2).impulse(T = t_oto), (t_oto, impulse_oto))
        
        t_scc = numpy.arange(0, 0.03, 0.001)
        impulse_scc = [-6305.60000000000, -5163.09582724642, -4227.69249097475, -3461.84897852112, -2834.82930831595, -2321.46898702215, -1901.16507005904, -1557.04929309447, -1275.31108936862, -1044.64332315783, -855.788494715896, -701.167204303977, -574.573964266030, -470.928151020200, -386.070101795209, -316.594172757778, -259.712058567572, -213.140887893487, -175.011603771427, -143.793951785632, -118.235065585886, -97.3091849617711, -80.1764884837337, -66.1493885203042, -54.6649359321624, -45.2622269465280, -37.5639054704159, -31.2610184656376, -26.1006165785713, -21.8756023944357]
        
        numpy.testing.assert_almost_equal(o.getTf(3, 3).impulse(T = t_scc), (t_scc, impulse_scc))
        numpy.testing.assert_almost_equal(o.getTf(4, 4).impulse(T = t_scc), (t_scc, impulse_scc))
        numpy.testing.assert_almost_equal(o.getTf(5, 5).impulse(T = t_scc), (t_scc, impulse_scc))
        
        
    def test_FindSteadyState(self):
        o = OrmsbyModel()
        u = o.input.expr(0)
        u['omega'] = numpy.array([0.1, 0.2, 0.3])
        u['f'] = numpy.array([0.4, 0.5, 0.6])
        
        x_s = o.state.expr(o.FindSteadyState(u))
        y_s = cs.mtimes(o.C, x_s) + cs.mtimes(o.D, u)

        numpy.testing.assert_allclose(cs.mtimes(o.A, x_s) + cs.mtimes(o.B, u), numpy.zeros(x_s.shape), atol=1e-16)
        numpy.testing.assert_allclose(y_s, numpy.atleast_2d([1.83486, 2.29358, 2.75229, 0, 0, 0]).T, atol=1e-5)


if __name__ == "__main__":
    unittest.main()