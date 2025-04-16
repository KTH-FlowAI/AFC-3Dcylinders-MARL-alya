#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Cylinder 3D parameters
#
# Just copy this code to the main folder and rename it
# to parameters.py to use it
#
# Pol Suarez, Fran Alcantara, Arnau Miro
from __future__ import print_function, division

import numpy as np
import math
import os

from jets import build_jets, JetCylinder

### CASE NAME ************************************************

case            = 'cylinder'
simu_name       = '3DCyl'
dimension       =  3
reward_function = 'drag_plain_lift' # at this moment, the only one used and parametrized

### **********************************************************
### DOMAIN BOX ***********************************************

radius = 0.5 # cylinder_radius
D      = 2*radius
length = 30*D 
width  = 15*D
Lz     = 2*np.pi

cylinder_coordinates   = [width*0.5, width*0.5, 0.0]
xkarman                = 3 * int(2 * cylinder_coordinates[0])

Parabolic_max_velocity = 1.5

### **********************************************************
### MULTI-ENVIRONMENT SETUP **********************************

nb_proc       = 50 #Number of calculation processors
num_servers   = 5 # number of environment in parallel

proc_per_node = 200 # in DARDEL
#proc_per_node = int(os.getenv('SLURM_NTASKS_PER_NODE'))*int(os.getenv('SLURM_CPUS_PER_TASK'))

mem_per_node  = 220000 # MB RAM in each node
#mem_per_node   = int(os.getenv('SLURM_MEM_PER_NODE'))

mem_per_cpu    = mem_per_node//proc_per_node
#mem_per_cpu   = int(os.getenv('SLURM_MEM_PER_CPU'))

mem_per_srun  = int(mem_per_node//(proc_per_node//nb_proc)) # partition in memory allocation

print("POOOOOOOL ---> SLURM_MEM_PER_NODE = %d" % mem_per_node)
print("POOOOOOOL ---> SLURM_MEM_PER_CPU  = %d" % mem_per_cpu)
print("POOOOOOOL ---> SLURM_MEM_PER_SRUN  = %d" % mem_per_srun)

num_episodes  = 1000 # Total number of episodes
nb_actuations = 80 # Number of actuation of the neural network for each episode
nb_actuations_deterministic = nb_actuations*4
batch_size    = 20

if num_servers==1:
	sync_episodes=False
else: 
	sync_episodes=True

### *****************************************************
### RUN BASELINE **************************************** 

run_baseline = False

### *****************************************************
### TIMESCALES ******************************************

baseline_duration   = 100.0 # to converge with velocity max = 1    
baseline_time_start = 0.0

delta_t_smooth   = 0.25           # ACTION DURATION smooth law duration
delta_t_converge = 0.0            # Total time that the DRL waits before executing a new action
smooth_func      = 'EXPONENTIAL'  # 'LINEAR', 'EXPONENTIAL', 'CUBIC' (# TODO: cubic is still not coded)
short_spacetime_func = False       # override smooth func --> TODO: need to fix string size --> FIXED in def_kintyp_functions.f90 (waiting for merging)

### *****************************************************
### FLUID PROPERTIES ************************************

mu  = 1E-2
rho = 1.0

### *****************************************************
### POSTPROCESS OPTIONS *********************************

norm_reward = 1.25 # like PRESS, try to be between -1,1    
penal_cl    = 0.6  # avoid asymmetrical strategies
norm_press  = 2.0 
time_avg    = 0.1 #corresponds to the last Tk
post_process_steps = 50  # TODO: put this into a include

### *****************************************************
### JET SETUP *******************************************

norm_Q = 0.088 # (0.088/2)/5 asa said in papers, limited Q for no momentum or discontinuities in the CFD solver

#location jet over the cylinder 0 is top centre
jet_angle = 0

nz_Qs = 6 ## DEBUG param --> define how many Qs to control in the span (needs to define Q profile)

## it will place many slices of witness as Qs locations we have 

Qs_position_z = [] 
for nq in range(nz_Qs): Qs_position_z.append((Lz/(nz_Qs+1))*(nq+1))
print("POOOOOOOL  Qs position -->",Qs_position_z)

delta_Q_z = Lz/(nz_Qs+1)


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


### ****************************************************
### BL option? ****************************************************

boundary_layer         = False #TODO: For what is this used? It is imported in geo_file_maker.py

if boundary_layer:
	outer_radius       = 1.3*radius + dp_left # Boundary Layer radius
	Transfinite_number = 20
	Progression_number = 1.025


### ****************************************************
### STATE OBSSERVATION -- WITNESS MAP ******************

## HERE WE HAVE 3 CHOICES TO LOCATE PROBES: 1-- free (now 24 probes) // 2-- jean's 151 probes // 3-- 5 probes

positions_probes_for_grid_z = Qs_position_z # beta --> for now, coincide with peaks Qs jets
probes_location      = 2
list_position_probes = []

############################################################################################
####    These are the probe positions from Jean paper (151 probes) BUT EXTRUDED IN 3D   ####
############################################################################################

if probes_location == 2:
	list_radius_around = [radius + 0.2, radius + 0.5]
	list_angles_around = np.arange(0, 360, 10)
	#positions_probes_for_grid_z = [2.5]  #beta

	for crrt_radius in list_radius_around:
		for crrt_angle in list_angles_around:
			for crrt_z in positions_probes_for_grid_z:
				angle_rad = np.pi * crrt_angle / 180.0
				list_position_probes.append(np.array([crrt_radius * math.cos(angle_rad) + cylinder_coordinates[0], crrt_radius * math.sin(angle_rad) + cylinder_coordinates[1], crrt_z]))

	positions_probes_for_grid_x = [0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
	positions_probes_for_grid_y = [-1.5, -1, -0.5, 0.0, 0.5, 1, 1.5]

	for crrt_x in positions_probes_for_grid_x:
		for crrt_y in positions_probes_for_grid_y:
			for crrt_z in positions_probes_for_grid_z:
				list_position_probes.append(np.array([crrt_x + cylinder_coordinates[0], crrt_y + cylinder_coordinates[1], crrt_z]))

	positions_probes_for_grid_x = [-0.25, 0.0, 0.25, 0.5]
	positions_probes_for_grid_y = [-1.5, -1, 1, 1.5]

	for crrt_x in positions_probes_for_grid_x:
		for crrt_y in positions_probes_for_grid_y:
			for crrt_z in positions_probes_for_grid_z:
				list_position_probes.append(np.array([crrt_x + cylinder_coordinates[0], crrt_y + cylinder_coordinates[1], crrt_z]))

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

#####################################################################################
####   These are the probe positions from Jean paper with only 5 probes in 3D    ####
#####################################################################################

if probes_location == 3:
	list_radius_around = [radius + 0.2]
	list_angles_around = np.arange(90, 271, 180)
	list_in_z          = [1.0, 2.5, 3.0]

	for crrt_radius in list_radius_around:
		for crrt_angle in list_angles_around:
			for crrt_z in list_in_z:
				angle_rad = np.pi * crrt_angle / 180.0
				list_position_probes.append(np.array([crrt_radius * math.cos(angle_rad) + cylinder_coordinates[0], crrt_radius * math.sin(angle_rad) + cylinder_coordinates[1], crrt_z]))

	positions_probes_for_grid_x = [0.85]
	positions_probes_for_grid_y = [-1, 0.0, 1]

	for crrt_x in positions_probes_for_grid_x:
		for crrt_y in positions_probes_for_grid_y:
			for crrt_z in list_in_z:
				list_position_probes.append(np.array([crrt_x + cylinder_coordinates[0], crrt_y + cylinder_coordinates[1],crrt_z]))

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
	'simulation_duration':  baseline_duration,      #the time the simulation is in permanant regime
	'simulation_timeframe': [baseline_time_start,baseline_time_start+baseline_duration],
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
	'bound':           [5], # Boundaries to postprocess. Comma separeted
	'porous':          False, # Variational boundaries to postprocess. Comma separeted 
	'density':         rho, # Fluid density #TODO: repeated parameter
	'veloc':           1, # average velocity of a parabolic inflow
	"scale_area":      6.28, # Projected frontal area. Scale if it is need it
	"d":               0, # Distance for momentum calculation
	"time":            -time_avg, # Time to average. Negative to average from last time, 0 to average over total time, positive to postprocess form initial time
	"initial_time":    None, # initial_time required only if positive "time averaging" used
	"rotation_vector": [1,0,0], # Vector rotation in case rotation between axis is needed
	"phi":             90, # Rotation angle
	"D_exp":           1.42, # SCHAFER Experimental Drag
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
	"norm_Q":                        norm_Q, # (0.088/2)/5 asa said in papers, limited Q for no momentum or discontinuities in the CFD solver
	"norm_reward":                   norm_reward, # like PRESS, try to be between -1,1  
	"penal_cl":                      penal_cl,  # avoid asymmetrical strategies
	"norm_press":                    norm_press,  # force the PRESS comms to be between -1,1 (seggestion for better performance DRL)
	"offset_reward":                 1.42, # start the reward from 0
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
