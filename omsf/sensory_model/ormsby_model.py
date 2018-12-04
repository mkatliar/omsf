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
import casadi as cs
import casadi_extras as ct

import numpy as np
import scipy.linalg as la
import control.matlab

from omsf import dsys, util
from omsf.gravity import DEFAULT_GRAVITY
from omsf.signal import INERTIAL_SIGNAL


class OrmsbyModel(dsys.LTISystem):
    '''
    Vestibular sensory dynamics model by Ormsby.

    Input: visual input x9, linear acceleration x3 (m/s^2), rotational velocity x3 (rad/s)
    Output: otolith neurons firing rate x3, (1/s), SCC neurons firing rate x3 (1/s).
    '''

    def __init__(self):
        '''Constructor
        '''

        tau_A = 30
        tau_L = 18
        tau_S = 0.005
        tau_R = 0.01
        HF = 6300
        
        H_scc = control.matlab.tf([HF * tau_L * tau_S, 0], np.polymul([tau_L, 1], [tau_S, 1])) \
            * control.matlab.tf(np.polymul([tau_A, 0], [tau_R, 1]), [tau_A, 1])
        H_oto = 90 * control.matlab.tf([1, 0.1], [1, 0.2]) / np.linalg.norm(DEFAULT_GRAVITY)
        
        '''
        H_scc_num = H_scc.num[0][0]
        H_scc_den = H_scc.den[0][0]
        H_oto_num = H_oto.num[0][0]
        H_oto_den = H_oto.den[0][0]
        
        H_gross_num = [
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], H_oto_num, [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_oto_num, [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_oto_num, [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_scc_num, [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_scc_num, [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_scc_num]
        ]
        
        H_gross_den = [
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], H_oto_den, [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_oto_den, [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_oto_den, [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_scc_den, [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_scc_den, [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], H_scc_den]
        ]
        '''
        ss_scc = control.matlab.tf2ss(H_scc)
        ss_oto = control.matlab.tf2ss(H_oto)
        
        output_struct = ct.struct_symMX([
            ct.entry('fr_oto', shape=3),
            ct.entry('fr_scc', shape=3)
            ])
        
        state_struct = ct.struct_symMX([
            ct.entry('oto', shape=ss_oto.states, repeat=3),
            ct.entry('scc', shape=ss_scc.states, repeat=3)
            ])
        
        input_struct = ct.struct_MX([
            ct.entry('f', expr=INERTIAL_SIGNAL['f']),
            ct.entry('omega', expr=INERTIAL_SIGNAL['omega'])
        ])
        
        A = la.block_diag(ss_oto.A, ss_oto.A, ss_oto.A, ss_scc.A, ss_scc.A, ss_scc.A)
        B = la.block_diag(ss_oto.B, ss_oto.B, ss_oto.B, ss_scc.B, ss_scc.B, ss_scc.B)
        C = la.block_diag(ss_oto.C, ss_oto.C, ss_oto.C, ss_scc.C, ss_scc.C, ss_scc.C)
        D = la.block_diag(ss_oto.D, ss_oto.D, ss_oto.D, ss_scc.D, ss_scc.D, ss_scc.D)
        
        A = cs.sparsify(A)
        B = cs.sparsify(B)
        C = cs.sparsify(C)
        D = cs.sparsify(D)
        
        dsys.LTISystem.__init__(self, A, B, C, D, input_struct, state_struct, output_struct)
		
        '''
        state_struct = ct.struct_symMX([
            ct.entry('oto', shape=ss_oto.states, repeat=3),
            ct.entry('scc', shape=ss_scc.states, repeat=3)
        ])

        ode = ct.struct_MX(state_struct)        
        ode['oto'] = [cs.mtimes(ss_oto.A, x) + cs.mtimes(ss_oto.B, u) for x, u in zip(state_struct['oto'], INERTIAL_SIGNAL['f'])]
        ode['scc'] = [cs.mtimes(ss_scc.A, x) + cs.mtimes(ss_scc.B, u) for x, u in zip(state_struct['scc'], INERTIAL_SIGNAL['omega'])]

        output_struct = ct.struct_MX([
            ct.entry('fr_oto', expr=[cs.mtimes(ss_oto.C, x) + cs.mtimes(ss_oto.D, u) for x, u in zip(state_struct['oto'], INERTIAL_SIGNAL['f'])]),
            ct.entry('fr_scc', expr=[cs.mtimes(ss_scc.C, x) + cs.mtimes(ss_scc.D, u) for x, u in zip(state_struct['scc'], INERTIAL_SIGNAL['omega'])])
        ])

        self.state = state_struct
        self.output = output_struct
        self.ode = ode
        '''
        
        self.outputName = ['FR f_x [Hz]', 'FR f_y [Hz]', 'FR f_z [Hz]',
            'FR \omega_x [Hz]', 'FR \omega_y [Hz]', 'FR \omega_z [Hz]']
            
