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
import math

from jets import build_jets, JetCylinder


case            = 'cylinder'
simu_name       = '2DCyl'
dimension       = '2D'
reward_function = 'drag_plain_lift' # at this moment, the only one used and parametrized


## Geometry and dp section
radius = 0.5 # cylinder_radius
D      = 2*radius
length = 22*D 
width  = 4*D+0.1

dp_left   = length/115
dp_right  = dp_left*6
dp_cyl    = dp_left/25
dp_karman = (dp_left + 0.6*dp_right) / 2.8

nb_proc     = 1 #Number of calculation processors
num_servers = 6 # number of environment in parallel

# Simulation params
simulation_duration   = 25.0 # to converge with velocity max = 1    
simulation_time_start = 0.0

delta_t_smooth   = 0.25      # smooth law duration
delta_t_converge = 0.0       # Total time that the DRL waits before executing a new action
smooth_func      = 'linear'  # 'linear', 'parabolic', 'cubic' (# TODO: cubic is still not coded)

# fluid properties
mu  = 1E-2
rho = 1.0

# post options
post_process_steps = 200  # TODO: put this into a include

## Domain section
Dict_domain ={
	"downleft":  [0,0],
	"downright": [length,0],
	"upright":   [length,width],
	"upleft":    [0,width],
}

cylinder_coordinates   = [2*D,2*D]
xkarman                = 3 * int(2 * cylinder_coordinates[0])
Parabolic_max_velocity = 1.5
boundary_layer         = False #TODO: For what is this used? It is imported in geo_file_maker.py

if boundary_layer:
	outer_radius       = 1.3*radius + dp_left # Boundary Layer radius
	Transfinite_number = 20
	Progression_number = 1.025

if dimension == '3D': # TODO: This is imported in geo_file_maker.py
	height        = dp_right
	Layers_number = 1 # Number of layers for 3D case // 1 for 2D OpenFOAM


## DRL Section
num_episodes = 500 # Total number of episodes

#location jet over the cylinder 0 is top centre
jet_angle = 0

jets_definition = {
	'JET_TOP' : {
		'width':           10,
		'radius':          radius,
		'angle' :          jet_angle,
		'positions_angle': 90+jet_angle, # make sure the width doesn't not coincide with 0,90,180 or 270
		'positions':       [cylinder_coordinates[0],cylinder_coordinates[1]+radius],
		'remesh':          False
	},
	'JET_BOTTOM' : {
		'width':           10,
		'radius':          radius,
		'angle' :          jet_angle,
		'positions_angle': 270-jet_angle, # make sure the width doesn't not coincide with 0,90,180 or 270
		'positions':       [cylinder_coordinates[0],cylinder_coordinates[1]-radius],
		'remesh':          False
	}
}

# Build the jets
jets   = build_jets(JetCylinder,jets_definition,delta_t_smooth)
n_jets = len(jets)

geometry_params = { # Kept for legacy purposes but to be deleted when reworking the mesh script
	'output':              '.'.join(["cylinder", 'geo']),
	'jet_width':           10,
	'jet_angle' :          jet_angle,
	'jet_name' :           ['JET_TOP','JET_BOTTOM'],
	'jet_positions_angle': [90+jet_angle, 270-jet_angle], # make sure the width doesn't not coincide with 0,90,180 or 270
	'jet_positions':       [[cylinder_coordinates[0],cylinder_coordinates[1]+radius],[cylinder_coordinates[0],cylinder_coordinates[1]-radius]],
	'remesh':              False
}
assert jet_angle != geometry_params["jet_width"]/2 # Maybe to check during mesh construction?

# Environment
nb_actuations = 5 # Number of actuation of the neural network for each episode
nb_actuations_deterministic = nb_actuations*4

## HERE WE HAVE 3 CHOICES TO LOCATE PROBES: 1-- free (now 24 probes) // 2-- jean's 151 probes // 3-- 5 probes
probes_location      = 2
list_position_probes = []

##############################################################
#####   VARIABLE PROBES LOCATIONS (this has 24 probes)   #####
##############################################################
if probes_location == 1:
	# left positions probes
	left_position_probes     = []
	left_numbers_of_layers_x = 0
	left_numbers_of_layers_y = 0

	left_X_probes = 1.8*(1-np.exp(-np.linspace(
		start = cylinder_coordinates[0]/4,
		stop  = cylinder_coordinates[0]/1.5,
		num   = left_numbers_of_layers_x))
	)
	left_Y_probes = np.linspace(
		start = cylinder_coordinates[1]-D*0.9,
		stop  = cylinder_coordinates[1]+D*0.9,
		num   = left_numbers_of_layers_y
	)

	for X in left_X_probes:
		for Y in left_Y_probes:
			left_position_probes.append([X,Y])

	# cylinder positions probes
	cylinder_position_probes = []

	cylinder_numbers_of_layers_theta        = 3
	cylinder_numbers_of_layers_r            = 2
	cylinder_numbers_of_layers_karman_theta = 3
	cylinder_numbers_of_layers_karman_r     = 4

	delta              = np.abs(geometry_params["jet_width"])*5
	thetas_up_probes   = np.linspace(start = 90+jet_angle-delta*0.4, stop = 90+jet_angle+delta*0.2, num = cylinder_numbers_of_layers_theta)*np.pi/180
	thetas_down_probes = np.linspace(start = 270-jet_angle-delta*0.2, stop = 270-jet_angle+delta*0.4, num = cylinder_numbers_of_layers_theta)*np.pi/180
	thetas_karman      = np.linspace(start = -delta*1.2, stop = delta*1.2, num = cylinder_numbers_of_layers_karman_theta)*np.pi/180

	r_probes = (np.exp(np.linspace(
		start = 0,
		stop  = dp_cyl*30,
		num   = cylinder_numbers_of_layers_r
	)) -1) + radius + 5*dp_cyl

	r_karman_probes = (np.exp(np.linspace(
		start = 0,
		stop  = 0.5,
		num   = cylinder_numbers_of_layers_karman_r
	))*3 -3) + radius + 5*dp_cyl

	for r in r_probes:
		for theta in thetas_up_probes:
			cylinder_position_probes.append([r*np.cos(theta)+cylinder_coordinates[0],r*np.sin(theta)+cylinder_coordinates[1]])
		for theta in thetas_down_probes:
			cylinder_position_probes.append([r*np.cos(theta)+cylinder_coordinates[0],r*np.sin(theta)+cylinder_coordinates[1]])

	karman_position_probes = []
	theta_step_factor      = 0.83
	for r in r_karman_probes:
		theta_step_factor *= 0.87
	for theta in thetas_karman:
		theta *= theta_step_factor
	karman_position_probes.append([r*np.cos(theta)+cylinder_coordinates[0],r*np.sin(theta)+cylinder_coordinates[1]])

	list_position_probes += left_position_probes + cylinder_position_probes + karman_position_probes

	len_left_positions_probes    = len(left_position_probes)
	len_cylinder_position_probes = len(cylinder_position_probes)
	len_karman_position_probes   = len(karman_position_probes)

	cylinder_range_min = len_left_positions_probes + 1
	cylinder_range_max = len_left_positions_probes + len_cylinder_position_probes

	karman_range_min = cylinder_range_max + 1
	karman_range_max = cylinder_range_max + len_karman_position_probes
	tag_probs = {
		'left':     [1,len_left_positions_probes],
		'cylinder': [cylinder_range_min,cylinder_range_max],
		'karman':   [karman_range_min,karman_range_max]
	}

	output_params = {
		'locations':  list_position_probes,
		'tag_probs':  tag_probs,
		'probe_type': 'pressure'
	}

##########################################################################
####    These are the probe positions from Jean paper (151 probes)    ####
##########################################################################
if probes_location == 2:
	list_radius_around = [radius + 0.2, radius + 0.5]
	list_angles_around = np.arange(0, 360, 10)

	for crrt_radius in list_radius_around:
		for crrt_angle in list_angles_around:
			angle_rad = np.pi * crrt_angle / 180.0
			list_position_probes.append(np.array([crrt_radius * math.cos(angle_rad) + cylinder_coordinates[0], crrt_radius * math.sin(angle_rad) + cylinder_coordinates[1]]))

	positions_probes_for_grid_x = [0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
	positions_probes_for_grid_y = [-1.5, -1, -0.5, 0.0, 0.5, 1, 1.5]

	for crrt_x in positions_probes_for_grid_x:
		for crrt_y in positions_probes_for_grid_y:
			list_position_probes.append(np.array([crrt_x + cylinder_coordinates[0], crrt_y + cylinder_coordinates[1]]))

	positions_probes_for_grid_x = [-0.25, 0.0, 0.25, 0.5]
	positions_probes_for_grid_y = [-1.5, -1, 1, 1.5]

	for crrt_x in positions_probes_for_grid_x:
		for crrt_y in positions_probes_for_grid_y:
			list_position_probes.append(np.array([crrt_x + cylinder_coordinates[0], crrt_y + cylinder_coordinates[1]]))

	len_left_positions_probes    = 0
	len_cylinder_position_probes = 72
	len_karman_position_probes   = 79

	cylinder_range_min = len_left_positions_probes + 1
	cylinder_range_max = len_left_positions_probes + len_cylinder_position_probes

	karman_range_min = cylinder_range_max + 1
	karman_range_max = cylinder_range_max + len_karman_position_probes
	tag_probs = {
		'left':     [1,len_left_positions_probes],
		'cylinder': [cylinder_range_min,cylinder_range_max],
		'karman':   [karman_range_min,karman_range_max]
	}

	output_params = {
		'locations': list_position_probes,
		'tag_probs':tag_probs,
		'probe_type':'pressure'
	}

###############################################################################
####   These are the probe positions from Jean paper with only 5 probes    ####
###############################################################################
if probes_location == 3:
	list_radius_around = [radius + 0.2]
	list_angles_around = np.arange(90, 271, 180)

	for crrt_radius in list_radius_around:
		for crrt_angle in list_angles_around:
			angle_rad = np.pi * crrt_angle / 180.0
			list_position_probes.append(np.array([crrt_radius * math.cos(angle_rad) + cylinder_coordinates[0], crrt_radius * math.sin(angle_rad) + cylinder_coordinates[1]]))

	positions_probes_for_grid_x = [0.85]
	positions_probes_for_grid_y = [-1, 0.0, 1]

	for crrt_x in positions_probes_for_grid_x:
		for crrt_y in positions_probes_for_grid_y:
			list_position_probes.append(np.array([crrt_x + cylinder_coordinates[0], crrt_y + cylinder_coordinates[1]]))

	len_left_positions_probes    = 0
	len_cylinder_position_probes = 2
	len_karman_position_probes   = 3

	cylinder_range_min = len_left_positions_probes+1
	cylinder_range_max = len_left_positions_probes + len_cylinder_position_probes

	karman_range_min = cylinder_range_max + 1
	karman_range_max = cylinder_range_max + len_karman_position_probes
	tag_probs = {
		'left':     [1,len_left_positions_probes],
		'cylinder': [cylinder_range_min,cylinder_range_max],
		'karman':   [karman_range_min,karman_range_max]
	}

	output_params = {
		'locations':  list_position_probes,
		'tag_probs':  tag_probs,
		'probe_type': 'pressure'
	}

###############################################################################
######################################################################

simulation_params = {
	'simulation_duration':  simulation_duration,#the time the simulation is in permanant regime
	'simulation_timeframe': [simulation_time_start,simulation_time_start+simulation_duration],
	'delta_t_smooth':       delta_t_smooth,
	'delta_t_converge':     delta_t_converge,
	'smooth_func':          smooth_func,
	# 'dt': dt,
	'mu':                   mu,
	'rho':                  rho,
	'post_process_steps' :  post_process_steps
}

# Variational input 
variational_input = {
	'filename':        'cylinder', # basename 
	'bound':           [3], # Boundaries to postprocess. Comma separeted
	'porous':          False, # Variational boundaries to postprocess. Comma separeted 
	'density':         rho, # Fluid density #TODO: repeated parameter
	'veloc':           Parabolic_max_velocity*(2/3), # average velocity of a parabolic inflow
	"scale_area":      1, # Projected frontal area. Scale if it is need it
	"d":               0, # Distance for momentum calculation
	"time":            -3.27, # Time to average. Negative to average from last time, 0 to average over total time, positive to postprocess form initial time
	"initial_time":    None, # initial_time required only if positive "time averaging" used
	"rotation_vector": [1,0,0], # Vector rotation in case rotation between axis is needed
	"phi":             90, # Rotation angle
	"D_exp":           3.29, # SCHAFER Experimental Drag
	"S_exp":           0, # Experimental Side
	"L_exp":           1.01, # SCHAFER Experimental Lift
	"R_exp":           0, # Experimental Roll
	"P_exp":           0, # Experimental Pitch
	"Y_exp":           0 # Experimental yaw
}

# Optimization 
optimization_params = {
	"num_steps_in_pressure_history": 1,
	"min_value_jet_MFR":             -1,
	"max_value_jet_MFR":             1,
	"norm_Q":                        0.088, # asa said in papers, limited Q for no momentum or discontinuities in the CFD solver
	"norm_reward":                   1.25, # like PRESS, try to be between -1,1  
	"penal_cl":                      1,  # avoid asymmetrical strategies
	"norm_press":                    2,  # force the PRESS comms to be between -1,1 (seggestion for better performance DRL)
	"offset_reward":                 3.19, # start the reward from 0
	"avg_TIME_PROBE":                0.25, # % del Tk to avg probe comms 
	"zero_net_Qs":                   False,
	"random_start":                  False
}

inspection_params = {
	"plot":                False,  #TODO: inspection_params is never used
	"step":                50,
	"dump":                100,
	"range_pressure_plot": [-2.0, 1],
	"range_drag_plot":     [-0.175, -0.13],
	"range_lift_plot":     [-0.2, +0.2],
	"line_drag":           -0.1595,
	"line_lift":           0,
	"show_all_at_reset":   True,
	"single_run":          False
}
