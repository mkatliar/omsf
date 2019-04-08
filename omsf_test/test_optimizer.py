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
Created on Jan 17, 2015

@author: mkatliar
'''
import omsf, numpy as np, casadi as cs
import casadi_extras as ct
from casadi_extras import Pdq
import omsf.io.matlab
from motion_platform_x import MotionPlatformX

import os
import unittest

DATA_DIR = 'omsf_test/data'


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        platform = MotionPlatformX()

        #----------------------------------
        # The cost function.
        #----------------------------------
        w = cs.diag([1, 1, 1, 10, 10, 10, 0, 0, 0]) # Weighting matrix
        W = cs.mtimes(w.T, w)
        L = 0.01 * cs.sumsqr(platform.input.expr) \
            + 0.01 * cs.sumsqr(platform.state.expr) \
            + cs.mtimes([cs.transpose(omsf.INERTIAL_SIGNAL_ERROR), W, omsf.INERTIAL_SIGNAL_ERROR])

        scenario = omsf.Scenario(platform, lagrange_term=L)
        scenario.state.nominal['platform'] = platform.state.expr(0)

        data = omsf.io.matlab.loadRecordedMotion(os.path.join(DATA_DIR, 'AccDec.mat'))
        si = omsf.recordedMotionToSensorySignal(data, np.array([0, 0, -1]))

        self._scenario = scenario
        self._sensorySignal = [si.cut(3, 5)]
        self._platform = platform


    def test_optimize(self):
        optimizer = omsf.Optimizer()
        optimizer.optimizationOptions = {'ipopt': {'linear_solver': 'ma86', 'max_iter': 100}}
        
        res = optimizer.optimize(self._scenario, self._sensorySignal)
        self.assertTrue('trajectory' in res)
        self.assertEqual(len(res['trajectory']), len(self._sensorySignal))
        self.assertTrue('param' in res)
        self.assertTrue('objective' in res)
        self.assertTrue('stats' in res)


    def test_optimizeMaximumIterationsReached(self):
        '''Test than an OptimizerError is raised when maximum number of iterations is reached.
        '''
        optimizer = omsf.Optimizer()
        optimizer.optimizationOptions = {'ipopt': {'linear_solver': 'ma86', 'max_iter': 10}}
        
        with self.assertRaises(omsf.OptimizerError):
            optimizer.optimize(self._scenario, self._sensorySignal)

    
    def test_initialAndFinalStateBounds(self):
        
        # Set initial and final state bounds to 0.
        lb = cs.vertcat(self._platform.state.expr(0), self._platform.state.expr(0))
        ub = cs.vertcat(self._platform.state.expr(0), self._platform.state.expr(0))
        self._scenario.terminalConstraint = ct.Inequality(
            expr=cs.vertcat(self._scenario.initialState['platform'], self._scenario.finalState['platform']),
            lb=lb, ub=ub
        )

        optimizer = omsf.Optimizer()
        optimizer.optimizationOptions = {'ipopt': {'linear_solver': 'ma86', 'max_iter': 100}}
        
        res = optimizer.optimize(self._scenario, self._sensorySignal)
        [pm] = res['trajectory']
        
        # Check that initial and final states are within the specified limits
        t = pm.time
        state = pm.platformState
        self.assertTrue(np.all(lb - 1e-10 <= cs.vertcat(state(t[0]), state(t[-1]))))
        self.assertTrue(np.all(ub + 1e-10 >= cs.vertcat(state(t[0]), state(t[-1]))))


    @unittest.skip("Scenario.simulate() does not work and may be deprecated in future.")
    def test_simulate(self):
        platform = MotionPlatformX()
        scenario = omsf.Scenario(platform, lagrange_term=0)
        
        Nt = 1
        t = np.array([0, 1])        
        a_x = np.array([[1]])
        x = np.array([[0, 0], [0.5, 1]]).T
        z = np.zeros((0, Nt))

        mot = ct.SystemTrajectory(x=x, z=z, u=a_x, pdq=Pdq(t, poly_order=1))
        
        ves = scenario.simulate(mot)
        
        expected_f = np.zeros((3, Nt))
        expected_f[0, :] = -a_x
        expected_f += np.atleast_2d(scenario.gravity).T
        expected_omega = np.zeros((3, Nt))
        expected_alpha = np.zeros((3, Nt))

        np.testing.assert_equal(ves, np.vstack((expected_f, expected_omega, expected_alpha)))


if __name__ == "__main__":
    unittest.main()