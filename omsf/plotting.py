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
import numpy as np

#from omsf.gravity import DEFAULT_GRAVITY

def _plot_components(fig, t, y, ttl, x_label, y_label, *args, **kwargs):
    """Plots each column of y on a separate subplot with a label defined by y_label. 

    If 6-th argument is a function handle, it is interpreted as a
    function to use for plotting (i.e., plot(), hist()). By default, plot() is used.
    """

    assert t.ndim == 1
    
    if y.shape[0] != len(t) and y.shape[1] == len(t):
        y = y.T

    n_comp = y.shape[1]
    n_rows = n_comp // 3
    
    if not isinstance(y_label, list):
        y_label = [y_label for _ in range(n_comp)]
    
    if not isinstance(x_label, list):
        x_label = [x_label for _ in range(n_comp)]

    ax = []
    
    for i in range(n_comp):
        j = i % 3
        row = i // 3

        a = fig.add_subplot(n_rows, 3, i + 1, sharey=ax[-1] if j > 0 else None)
        
        plot_function = kwargs['plot_function'] if 'plot_function' in kwargs else a.plot
        plot_function(t, y[:, i], *args, **kwargs);            
        
        if row + 1 == n_rows:
            a.set_xlabel(x_label[i])
            
        a.set_ylabel(y_label[i])
    
        a.grid(True)

        ax.append(a)

    fig.suptitle(ttl)
    
        
def _plot_vestibular_input(fig, t, ves, *args, **kwargs):
    y_label = [r'$f_x$ [$m/s^2$]', r'$f_y$ [$m/s^2$]', r'$f_z$+g [$m/s^2$]',
        r'$\omega_x$ [$s^{-1}$]', r'$\omega_y$ [$s^{-1}$]', r'$\omega_z$ [$s^{-1}$]',
        r'$\alpha_x$ [$s^{-2}$]', r'$\alpha_y$ [$s^{-2}$]', r'$\alpha_z$ [$s^{-2}$]']
    
    ves1 = np.array(ves)
    #ves1[2, :] += np.norm(DEFAULT_GRAVITY)
    ves1[2, :] += 9.81
    
    _plot_components(fig, t, ves1, 'Inertial signal', 't [s]', y_label, *args, **kwargs)


def plotRecordedMotion(motion, fig, **kwargs):
    label = ['$a_x [m/s^2]$', '$a_y [m/s^2]$', '$a_z [m/s^2]$', \
            '$\omega_x [s^{-1}]$', '$\omega_y [s^{-1}]$', '$\omega_z [s^{-1}]$', \
            '$\omega\'_x [s^{-2}]$', '$\omega\'_y [s^{-2}]$', '$\omega\'_z [s^{-2}]$']
        
    _plot_components(fig, motion.time, np.vstack([motion.gia, motion.rotationalVelocity, motion.rotationalAcceleration]), '', '$t [s]$', label, **kwargs)

    
def PlotVestibularInput(sensory_signal, fig, *args, **kwargs):
    """Plot vestibular part of sensory input.
    """
    _plot_vestibular_input(fig, sensory_signal.time, sensory_signal.getInertialInput(sensory_signal.time), *args, **kwargs)
        
    
