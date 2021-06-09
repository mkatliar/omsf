##
## Copyright (c) 2015-2021 Mikhail Katliar,
## Max Planck Institute for Biological Cybernetics & University of Freiburg.
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
# -*- coding: utf-8 -*-
"""
@author: mkatliar
"""

import casadi as cs
from omsf import transform


class PrismaticLink:
    '''Prismatic link class.
    '''

    def __init__(self, a, alpha, theta):
        self._a = a
        self._alpha = alpha
        self._theta = theta

        d = cs.SX.sym('d')
        self.LocalToBase = cs.Function('PrismaticLink_LocalToBase', [d],
            [transform.denavitHartenberg(self._a, d, self._alpha, self._theta)])


class RevoluteLink:
    '''Revolute link class.
    '''

    def __init__(self, a, d, alpha, theta_scale=1, theta_offset=0):
        self._a = a
        self._d = d
        self._alpha = alpha
        self._thetaScale = theta_scale
        self._thetaOffset = theta_offset
        self._TzRxTx = cs.mtimes([
            transform.translationZ(self._d),
            transform.rotationX(self._alpha),
            transform.translationX(self._a)
        ])

        theta = cs.SX.sym('theta')
        self.LocalToBase = cs.Function('RevoluteLink_LocalToBase', [theta],
            [transform.denavitHartenberg(self._a, self._d, self._alpha, self._thetaScale * theta + self._thetaOffset)])
