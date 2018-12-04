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
import numpy as np
import casadi as cs
from operator import itemgetter
from typing import NamedTuple

import casadi_extras as ct
from casadi_extras import CollocationScheme, cheb, Pdq
from casadi_extras.collocation import PiecewisePoly, PolynomialBasis, collocationPoints

from .gravity import GRAVITY


class Optimizer(object):
    '''Trajectory and parameters optimizer.
    '''

    def __init__(self):
        '''Constructor
        '''

        self.samplingTime = 0.05
        self.optimizationOptions    = {}
        self.parallelization = 'serial'
        self.nlpSolverPlugin = 'ipopt'  # Which NlpSolver plugin to use
        self.jit = False
        self.jitOptions = ['-O3']
        self.jitCompiler = 'gcc'

        # TODO: rename to collocationPolynomialOrder?
        self.numCollocationPoints = 5
        self.collocationMethod = 'legendre'
        self.iterationCallback = None


    def optimize(self, scenario, sensory_signal):
        '''Find optimal control input and parameters for the scenario.
        '''

        print('This is OMSF trajectory optimizer. jit is {0}, parallelization is set to {1}'
            .format('ON' if self.jit else 'OFF', self.parallelization))

        # Create DAE
        dae = scenario.makeDae()

        w = []
        g = []
        f = 0
        w0 = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        t_total = 0
        scheme = []
        t = []

        for ss in sensory_signal:        
            t_i = np.arange(ss.startTime, ss.stopTime, self.samplingTime)
            
            # Reference output
            y_ref = ss.inertialSignal
            
            # Visual input
            u_vis = ss.visualSignal
                    
            # Init solver
            print('Initializing NLP...')
            nlp_i, scheme_i = self._initNlp(dae, t_i, y_ref, u_vis, scenario)
            [w_i, g_i] = itemgetter('x', 'g')(nlp_i)
            
            w0_i, lbw_i, ubw_i, lbg_i, ubg_i = self._initBounds(w_i, g_i, scenario)

            # Append problem components to the list
            w.append(w_i)
            g.append(g_i)
            f += nlp_i['f']
            w0.append(w0_i)
            lbw.append(lbw_i)
            ubw.append(ubw_i)
            lbg.append(lbg_i)
            ubg.append(ubg_i)
            scheme.append(scheme_i)
            t.append(t_i)

            # Count total time
            t_total += t_i[-1] - t_i[0]

        # Divide integrated Lagrange term over total time
        f /= t_total

        # Adding constraints equaling parameters for all trajectories
        for i in range(len(scheme) - 1):
            g.append(scheme[i + 1].p[:, 0] - scheme[i].p[:, -1])
            lbg.append(np.zeros(dae.np))
            ubg.append(np.zeros(dae.np))
        
        print('Initializing NLP solver. Please be patient, this may take several minutes...')

        opts = self.optimizationOptions

        if self.iterationCallback is not None:
            iter_callback = IterationCallback('OmsfIterationCallback', 
                w=w, g=g, t=t, scheme=scheme, scenario=scenario, nested_callback=self.iterationCallback)
            opts['iteration_callback'] = iter_callback
        else:
            opts['iteration_callback'] = None

        nlp_solver = cs.nlpsol('OmsfNlpSolver', self.nlpSolverPlugin, 
            {'x': cs.vertcat(*w), 'f': f, 'g': cs.vertcat(*g)}, self.optimizationOptions)
        
        # Run the optimization.
        print('Starting optimization.')
        sol_out = nlp_solver(x0=cs.vertcat(*w0), lbg=cs.vertcat(*lbg), ubg=cs.vertcat(*ubg), 
            lbx=cs.vertcat(*lbw), ubx=cs.vertcat(*ubw))
        
        # Extract optimized trajectories
        nw = 0
        ng = 0
        res = []

        for w_i, g_i, scheme_i, t_i in zip(w, g, scheme, t):
            w_opt = w_i(sol_out['x'][nw : nw + w_i.numel()])                        
            res.append(_makeTrajectory(w_opt, t_i, scenario, scheme_i))

            nw += w_i.numel()
            ng += g_i.numel()

        # Extract optimized parameters
        par_opt = w_opt['p'][:, -1]
        
        return {'trajectory': res, 'param': par_opt, 'objective': sol_out['f']}


    def _initBounds(self, w, g, scenario):
        '''
        TODO: merge with _initNlp()
        '''

        lbg = g(-cs.inf)
        ubg = g(cs.inf)
        lbg["collocation"] = 0
        ubg["collocation"] = 0

        lbg['path'] = cs.repmat(scenario.constraint.lb, 1, lbg['path'].shape[1])
        ubg['path'] = cs.repmat(scenario.constraint.ub, 1, ubg['path'].shape[1])

        lbg['terminal'] = scenario.terminalConstraint.lb
        ubg['terminal'] = scenario.terminalConstraint.ub


        '''
        # Gravity XY-component tolerance.
        tol_gxy = 0.0

        if self.motionPlatform.initialStateConstraintStruct is not None:
            (lbg['cstate_init'], ubg['cstate_init']) = self.motionPlatform.initialStateConstraintBounds

        if self.initialPositionUpright:
            lbg["cupr", 0] = [-tol_gxy, -tol_gxy, -np.Inf]
            ubg["cupr", 0] = [ tol_gxy,  tol_gxy,  0 ]
        else:
            lbg["cupr", 0] = -np.Inf
            ubg["cupr", 0] =  np.Inf

        if self.finalPositionUpright:
            lbg["cupr", -1] = [-tol_gxy, -tol_gxy, -np.Inf]
            ubg["cupr", -1] = [ tol_gxy,  tol_gxy,  0 ]
        else:
            lbg["cupr", -1] = -np.Inf
            ubg["cupr", -1] =  np.Inf

        if self.initialStateSteady:
            lbg['csteady'] = 0
            ubg['csteady'] = 0
            lbg['csteady_alg'] = 0
            ubg['csteady_alg'] = 0
        else:
            lbg['csteady'] = -np.Inf
            ubg['csteady'] =  np.Inf
            lbg['csteady_alg'] = -np.Inf
            ubg['csteady_alg'] =  np.Inf
        '''

        ######################################################
        # Init bounds.
        ######################################################

        # Upper and lower bounds of the optimization variable.
        lbw = w(-np.Inf)
        ubw = w( np.Inf)

        lbw["x"] = cs.repmat(scenario.state.lb, 1, w['x'].shape[1])
        ubw['x'] = cs.repmat(scenario.state.ub, 1, w['x'].shape[1])
        lbw["Z"] = cs.repmat(scenario.algState.lb, 1, w['Z'].shape[1])
        ubw["Z"] = cs.repmat(scenario.algState.ub, 1, w['Z'].shape[1])
        lbw["u"] = cs.repmat(scenario.input.lb, 1, w['u'].shape[1])
        ubw["u"] = cs.repmat(scenario.input.ub, 1, w['u'].shape[1])
        lbw["p"] = cs.repmat(scenario.param.lb, 1, w['p'].shape[1])
        ubw["p"] = cs.repmat(scenario.param.ub, 1, w['p'].shape[1])

        # Initial approximation.
        w0 = w(0)
        w0['x'] = cs.repmat(scenario.state.nominal, 1, w['x'].shape[1])
        w0['Z'] = cs.repmat(scenario.algState.nominal, 1, w['Z'].shape[1])
        w0['u'] = cs.repmat(scenario.input.nominal, 1, w['u'].shape[1])
        w0['p'] = cs.repmat(scenario.param.nominal, 1, w['p'].shape[1])

        return w0, lbw, ubw, lbg, ubg


    def _initNlp(self, dae, t, y_ref, u_vis, scenario):
        '''
        Initialize NLP variables.

        @dae DAE model of platform + sensor
        @param t time vector
        @param y_ref reference inertial signal as a function of continuous time
        @param u_vis visual signal as a function of continuous time
        '''

        # Create time-dependent parameter function
        def tdp_fun(t):
            tdp = dae.tdp()
            tdp['y_ref'] = y_ref(t)
            tdp['u_vis'] = u_vis(t)
            
            return tdp

        # Create collocation scheme
        scheme = CollocationScheme(dae=dae,
            t=t, order=self.numCollocationPoints, method=self.collocationMethod, tdp_fun=tdp_fun, expand=True,
            parallelization=self.parallelization, repeat_param=True, 
            options={'jit': self.jit, 'compiler': self.jitCompiler, 'jit_options': self.jitOptions})

        # Optimization variable
        w = scheme.combine(['x', 'K', 'Z', 'u', 'p'])

        # Objective
        f = scheme.q[:, -1]
        
        # Construct constraint function
        # TODO: shall we support path constraints on sensor? Currently we don't.
        # TODO: scenario.input, scenario.param, scenario.constraint
        constraint = cs.Function('Scenario_constraint', 
            [scenario.state.expr, scenario.algState.expr, scenario.input.expr, scenario.param.expr, GRAVITY],
            [scenario.constraint.expr],
            ['x', 'z', 'u', 'p', 'g'], ['c']).expand()

        constraint_parallel = constraint.map('Scenario_constraint_parallel',
            self.parallelization, scheme.numTotalCollocationPoints, [4], [])

        # Construct upright constraints
        '''
        x0 = self._state(w['x0', :, 0])
        xT = self._state(w['xf', :, -1])
        z0 = self._algState(z[:,  0])
        zT = self._algState(z[:, -1])        
        g_entries.append(ct.entry("cupr", expr=[
            self.headFrameGravity(x0['platform'], z0['platform']), 
            self.headFrameGravity(xT['platform'], zT['platform'])
            ]))
        '''
        
        # Construct terminal constraint function
        terminal_constraint = cs.Function(
            'Scenario_terminalConstraint', 
            [scenario.initialState, scenario.finalState], [scenario.terminalConstraint.expr],
            ['x0', 'xf'], ['c']
        )
            
        # Make the NLP
        g = ct.struct_MX([
            ct.entry('collocation', expr=scheme.eq),
            ct.entry('path', expr=constraint_parallel(
                x=scheme.X, z=scheme.Z, u=scheme.U, p=scheme.p, g=scenario.gravity)['c']),
            ct.entry('terminal', expr=terminal_constraint(x0=scheme.x[:, 0], xf=scheme.x[:, -1])['c'])
        ])
            
        return {'x' : w, 'f' : f, 'g' : g}, scheme


def _makeTrajectory(w_opt, t, scenario, scheme):
    # NOTE: change to basis.numPoints if we don't use interval endpoints twice!
    basis = PolynomialBasis(scheme.butcher.c)
    M = basis.numPoints

    x_opt = w_opt['x']
    K_opt = w_opt['K']
    X_opt = scheme.evalX(x_opt, K_opt)
    Z_opt = w_opt['Z']
    u_opt = w_opt['u']
    p_opt = w_opt['p']

    x_platform = []
    x_sensor = []
    z_platform = []
    z_sensor = []
    u = []
    y = []

    for k in range(len(t) - 1):
        x = scenario.state.expr.parseMatrix(cs.horzcat(x_opt[:, k], X_opt[:, k * M : (k + 1) * M]))
        x_platform.append(x['platform'])
        x_sensor.append(x['sensor'])

        z = scenario.algState.expr.parseMatrix(Z_opt[:, k * M : (k + 1) * M])
        z_platform.append(z['platform'])
        z_sensor.append(z['sensor'])
        y.append(z['y_in'])

        u.append(u_opt[:, k])

    [quad] = cs.substitute([scheme.quad], 
        [scheme.x, scheme.K, scheme.Z, scheme.u, scheme.p],
        [x_opt, K_opt, Z_opt, u_opt, p_opt])
    quad = cs.horzsplit(cs.evalf(quad), np.arange(0, quad.shape[1] + 1, M))

    basis_x = PolynomialBasis(np.append(0, basis.tau))
    basis_z = basis
    basis_u = PolynomialBasis([0])

    return OptimizedTrajectory(
        platformState=PiecewisePoly(t, x_platform, basis_x),
        sensorState=PiecewisePoly(t, x_sensor, basis_x),
        platformAlgState=PiecewisePoly(t, z_platform, basis_z),
        sensorAlgState=PiecewisePoly(t, z_sensor, basis_z),
        input=PiecewisePoly(t, u, basis_u),
        inertialSignal=PiecewisePoly(t, y, basis_z),
        cost=PiecewisePoly(t, quad, basis_z),
        time=t
    )
    

class OptimizedTrajectory(NamedTuple):
    platformState: PiecewisePoly
    platformAlgState: PiecewisePoly
    sensorState: PiecewisePoly
    sensorAlgState: PiecewisePoly
    input: PiecewisePoly
    inertialSignal: PiecewisePoly
    cost: PiecewisePoly
    time: np.ndarray


class IterationCallback(cs.Callback):
    def __init__(self, name, w, g, t, scheme, scenario, nested_callback, opts={}):
        cs.Callback.__init__(self)

        self._w = w
        self._g = g
        self._t = t
        self._scheme = scheme
        self._scenario = scenario
        self._nestedCallback = nested_callback

        # Initialize internal objects
        self.construct(name, opts)

    
    def get_n_in(self): 
        return cs.nlpsol_n_out()


    def get_n_out(self): 
        return 1


    def get_name_in(self, i): 
        return cs.nlpsol_out(i)


    def get_name_out(self, i): 
        return "ret"


    def get_sparsity_in(self, i):
        n = cs.nlpsol_out(i)

        if n == 'f':
            return cs.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return cs.Sparsity.dense(cs.vertcat(*self._w).numel())
        elif n in ('g', 'lam_g'):
            return cs.Sparsity.dense(cs.vertcat(*self._g).numel())
        elif n == 'p':
            return cs.Spersity.dense(0)
        else:
            return cs.Sparsity(0, 0)


    def eval(self, arg):
        # Create dictionary
        sol_out = {}
        for (i, s) in enumerate(cs.nlpsol_out()): 
            sol_out[s] = arg[i]

        # Extract trajectories
        nw = 0
        ng = 0
        res = []

        for w_i, g_i, scheme_i, t_i in zip(self._w, self._g, self._scheme, self._t):
            w_opt = w_i(sol_out['x'][nw : nw + w_i.numel()])                        
            res.append(_makeTrajectory(w_opt, t_i, self._scenario, scheme_i))

            nw += w_i.numel()
            ng += g_i.numel()

        # Extract parameters
        par_opt = w_opt['p'][:, -1]
        
        # Call the nested callback
        self._nestedCallback(res, par_opt, sol_out['f'])

        return [0]
