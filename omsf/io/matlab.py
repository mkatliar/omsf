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
Created on Oct 21, 2016

@author: kotlyar
'''
import scipy.io
import omsf
    
def LoadSensoryInput(file_name):
    mat = scipy.io.loadmat(file_name)
    
    t = mat['time']
    t = t.squeeze() if t.size > 1 else t[0]
    
    return omsf.SensorySignal(t = t, 
                             v_v = mat['v_v'], omega_v = mat['omega_v'], g_v = mat['g_v'],
                             f_i = mat['f_i'], omega_i = mat['omega_i'], alpha_i = mat['alpha_i'])

def SaveSensoryInput(obj, file_name):
    scipy.io.savemat(file_name, {'v_v' : obj.v_v, 'omega_v' : obj.omega_v, 'g_v' : obj.g_v,
                                 'f_i' : obj.f_i, 'omega_i' : obj.omega_i, 'alpha_i' : obj.alpha_i,
                                 'time' : obj.time}, oned_as='column')

def loadRecordedMotion(file_name):
    mat = scipy.io.loadmat(file_name)
    
    t = mat['time']
    t = t.squeeze() if t.size > 1 else t[0]
        
    return omsf.RecordedMotion(t, v = mat['linearVelocity'], gia = mat['gia'], omega = mat['rotationalVelocity'], 
                          world_to_local = mat['worldToLocal'], omega_dot = mat['rotationalAcceleration'])

def SaveRecordedMotion(obj, file_name):
    scipy.io.savemat(file_name, {'linearVelocity' : obj.linearVelocity, 'worldToLocal' : obj.worldToLocal, \
                                 'rotationalVelocity' : obj.rotationalVelocity, 'rotationalAcceleration' : obj.rotationalAcceleration, \
                                 'gia' : obj.gia, 'time' : obj.time}, oned_as='column')