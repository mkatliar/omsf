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
class MotionLimits:
    
    def __init__(self, q_min, q_max, v_min, v_max, u_min, u_max):
        self.positionMin = q_min
        self.positionMax = q_max
        self.velocityMin = v_min
        self.velocityMax = v_max
        self.accelerationMin = u_min
        self.accelerationMax = u_max
        
    def __str__(self):
        return 'Position limits: {} velocity limits: {} acceleration limits: {}'.format(
            [self.positionMin, self.positionMax], [self.velocityMin, self.velocityMax], [self.accelerationMin, self.accelerationMax])