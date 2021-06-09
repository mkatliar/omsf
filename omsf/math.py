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
'''Math functions for OMSF

Created on Nov 29, 2015

@author: mkatliar
'''
import casadi as cs


def crossProductMatrix(omega):
    '''Cross-product matrix.

    Returns matrix Omega such as Omega * x = cross(omega, x) for all x.
    cross_product_matrix_arg is its inverse.
    See also http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication.
    '''
    return cs.vertcat(
        cs.horzcat(0, -omega[2], omega[1]),
        cs.horzcat(omega[2], 0, -omega[0]),
        cs.horzcat(-omega[1], omega[0], 0)
    )


def quatE(q):
    '''
    @param q quaternion format [w, x, y, z]
    '''
    return cs.vertcat(
        cs.horzcat(-q[1],  q[0], -q[3],  q[2]),
        cs.horzcat(-q[2],  q[3],  q[0], -q[1]),
        cs.horzcat(-q[3], -q[2],  q[1],  q[0])
    )


def quatG(q):
    '''
    @param q quaternion format [w, x, y, z]
    '''
    return cs.vertcat(
        cs.horzcat(-q[1],  q[0],  q[3], -q[2]),
        cs.horzcat(-q[2], -q[3],  q[0],  q[1]),
        cs.horzcat(-q[3],  q[2], -q[1],  q[0])
    )


def quatR(q):
    '''Rotation matrix from quaternion.

    @param q quaternion format [w, x, y, z]
    '''
    return cs.mtimes(quatE(q), quatG(q).T)


def quatRinv(q):
    '''Inverse rotation matrix from quaternion.

    @param q quaternion format [w, x, y, z]
    '''
    return cs.mtimes(quatG(q), quatE(q).T)


def vectorInterpolant(name, solver, grid, values, **kwargs):
    '''Analogous to casadi.interpolant(), but supports vector functions R^M -> R^N.

    @values is the list of points, each element being
    an array of function values correspoding to a single dimension.
    '''

    # Create an array of interpolants, 1 per dimension.
    scalar_interpolant = [cs.interpolant('{0}_{1}'.format(name, i), solver, grid, v, **kwargs) for i, v in enumerate(values)]

    # Combine scalar interpolants to a vector function.
    x = cs.MX.sym('x', len(grid))
    return cs.Function(name, [x], [cs.vertcat(*[f_s(x) for f_s in scalar_interpolant])])
