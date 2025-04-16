#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Bubble parameters
#
# Just copy this code to the main folder and rename it
# to parameters.py to use it
#
# Pol Suarez, Fran Alcantara, Arnau Miro
from __future__ import print_function, division

import numpy as np


raise ValueError('OLD PARAMETERS! WILL NOT WORK UNLESS FIXED')

dimension = '2D' # Available : 2D, 3D


h  = 1
Lx = 512*h
Ly = 2*h

if dimension == '3D': Lz = h
    
Dict_domain ={
	"downleft":    [0.0,0.0], #1
	"downright":   [Lx,0.0],  #2
	"upright":     [Lx,Ly],   #3
	"upleft":      [0.0,Ly],  #4
}

Parabolic_max_velocity = 1

dp = Lx/1028
    
Transfinite_number = 100
Progression_number = 1.001
    
x_start   = 30
x_end     = 265
x_rise    = 20
x_fall    = 115
d_restart = 70
d_rerise  = 100
x_restart = 2* (x_end-x_fall) - x_start - x_rise + d_restart
c         = 0.5
magnitude = 1

wall_velocity_conditions = {
	'x_start': x_start,
	'x_end': x_end,
	'x_rise': x_rise,
	'x_fall': x_fall,
	'd_rerise': d_rerise,
	'x_restart': x_restart,
	'c': c,
	'magnitude': magnitude
}