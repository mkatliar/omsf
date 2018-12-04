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
Created on Oct 21, 2015

@author: kotlyar
'''
import omsf
from omsf import transform
import casadi_extras as ct
import casadi as cs

class SurgePlatform(omsf.MotionPlatform):
    '''A very simple 1-DOF platform which can only do surge motion.
    '''
    
    def __init__(self):
        x = ct.struct_symMX([
            ct.entry('q'),
            ct.entry('v')
        ])

        u = cs.MX.sym('u')

        p = cs.MX.sym('pitch')
        
        # Init DAE function
        xdot = ct.struct_symMX(x)
        dae = cs.vertcat(
            xdot['q'] - x['v'],
            xdot['v'] - u
        )
                
        # PF -> WF transform
        T_PW = cs.mtimes(transform.translationX(x['q']), transform.rotationY(p))
        #T_PW = transform.translationX(x['q'])
        R_WP = transform.getRotationMatrix(transform.inv(T_PW))
        
        a_P = cs.vertcat(u, 0, 0)
        f_P = cs.mtimes(R_WP, omsf.GRAVITY) - a_P
        omega_P = cs.MX.zeros(3)
        alpha_P = cs.MX.zeros(3)

        # State bounds
        lbx = x()
        lbx['q'] = -1
        lbx['v'] = -1
        ubx = x()
        ubx['q'] = 1
        ubx['v'] = 1

        # Input bounds
        lbu = -1
        ubu = 1

        # Init base class
        # TODO: specify a_P instead of f_P and compute specific force due to gravity inside the MotionPlatform ctor? 
        omsf.MotionPlatform.__init__(self, state_derivative=xdot, 
            state=ct.Inequality(expr=x, lb=lbx, ub=ubx, nominal=x({'q': 0, 'v': 0})), 
            input=ct.Inequality(expr=u, lb=lbu, ub=ubu, nominal=0), 
            param=ct.Inequality(expr=p, lb=-cs.pi, ub=cs.pi, nominal=0),
            dae=dae, f_P=f_P, omega_P=omega_P, alpha_P=alpha_P, T_PW=T_PW)


'''
class MotionPlatformX(object):

    def __init__(self):

        # Define state
        x = ct.struct_symMX([
            ct.entry('x'),
            ct.entry('v')
        ])

        # Define alg state
        z = cs.MX.sym('z', 0)

        # Define input
        u = cs.MX.sym('u')

        # Define ODE
        ode = ct.struct_MX(x)
        ode['x'] = x['v']
        ode['v'] = u

        # Define Head->World rotation
        Rhw = omsf.transform.Identity()

        # Define linear acceleration
        a = cs.vertcat(u, 0, 0)

        # Define angular velocity
        omega = cs.MX.zeros(3)

        # Define angular acceleration
        alpha = cs.MX.zeros(3)

        self.x = x
        self.z = z
        self.u = u
        self.ode = ode
        self.Rhw = Rhw
        self.a = a
        self.omega = omega
        self.alpha = alpha

        
        #omsf.DoubleIntegratorMotionPlatform.__init__(self, motion_limits = [omsf.MotionLimits(-1, 1, -1, 1, -1, 1)])
'''