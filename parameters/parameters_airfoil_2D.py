#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Airfoil parameters
#
# Just copy this code to the main folder and rename it
# to parameters.py to use it
#
# Pol Suarez, Fran Alcantara, Arnau Miro
from __future__ import print_function, division

import os, math, numpy as np

from configuration import ALYA_PATH
from jets          import JetAirfoil, build_jets


# Define functions
def tj(ti, Pi, Pj, alpha):
	xi, yi = Pi
	xj, yj = Pj
	return ((xj-xi)**2 + (yj-yi)**2)**alpha + ti

def solve_problem(P0, P1, P2, P3, t0, t1, t2, t3, x_coordinate):
	'''
	Obtain the specific coordinate x in the spline by solving the cubic 
	equation
	'''
	x0 = P0[0]
	x1 = P1[0]
	x2 = P2[0]
	x3 = P3[0]

	b3 = - x0/((t2-t1)*(t2-t0)*(t1-t0)) + x1/((t2-t1)*(t2-t0)*(t1-t0)) + \
		x1/((t2-t1)*(t2-t0)*(t2-t1)) + x1/((t2-t1)*(t3-t1)*(t2-t1)) - \
		x2/((t2-t1)*(t2-t0)*(t2-t1)) - x2/((t2-t1)*(t3-t1)*(t2-t1)) - \
		x2/((t2-t1)*(t3-t1)*(t3-t2)) + x3/((t2-t1)*(t3-t1)*(t3-t2))
	b2 =   (t1+2*t2) *x0/((t2-t1)*(t2-t0)*(t1-t0)) - \
		(2*t2+t0) *x1/((t2-t1)*(t2-t0)*(t1-t0)) - \
		(2*t2+t0) *x1/((t2-t1)*(t2-t0)*(t2-t1)) - \
		(t1+t2+t3)*x1/((t2-t1)*(t3-t1)*(t2-t1)) + \
		(t0+t1+t2)*x2/((t2-t1)*(t2-t0)*(t2-t1)) + \
		(t3+2*t1) *x2/((t2-t1)*(t3-t1)*(t2-t1)) + \
		(t3+2*t1) *x2/((t2-t1)*(t3-t1)*(t3-t2)) - \
		(2*t1+t2) *x3/((t2-t1)*(t3-t1)*(t3-t2))
	b1 = - (2*t1*t2+t2*t2)    *x0/((t2-t1)*(t2-t0)*(t1-t0)) + \
		(t2*t2+2*t0*t2)    *x1/((t2-t1)*(t2-t0)*(t1-t0)) + \
		(t2*t2+2*t0*t2)    *x1/((t2-t1)*(t2-t0)*(t2-t1)) + \
		(t2*t3+t1*t2+t1*t3)*x1/((t2-t1)*(t3-t1)*(t2-t1)) - \
		(t0*t2+t1*t2+t0*t1)*x2/((t2-t1)*(t2-t0)*(t2-t1)) - \
		(2*t1*t3+t1*t1)    *x2/((t2-t1)*(t3-t1)*(t2-t1)) - \
		(2*t1*t3+t1*t1)    *x2/((t2-t1)*(t3-t1)*(t3-t2)) + \
		(t1*t1+2*t1*t2)    *x3/((t2-t1)*(t3-t1)*(t3-t2))
	b0 =   (t1*t2*t2)*x0/((t2-t1)*(t2-t0)*(t1-t0)) - \
		(t0*t2*t2)*x1/((t2-t1)*(t2-t0)*(t1-t0)) - \
		(t0*t2*t2)*x1/((t2-t1)*(t2-t0)*(t2-t1)) - \
		(t1*t2*t3)*x1/((t2-t1)*(t3-t1)*(t2-t1)) + \
		(t0*t1*t2)*x2/((t2-t1)*(t2-t0)*(t2-t1)) + \
		(t1*t1*t3)*x2/((t2-t1)*(t3-t1)*(t2-t1)) + \
		(t1*t1*t3)*x2/((t2-t1)*(t3-t1)*(t3-t2)) - \
		(t1*t1*t2)*x3/((t2-t1)*(t3-t1)*(t3-t2)) - \
		x_coordinate

	a2 = b2/b3
	a1 = b1/b3
	a0 = b0/b3

	P = - 1/3*a2*a2 + a1
	Q = - 2/27*a2*a2*a2 + a1*a2/3 - a0

	discriminant = P*P*P/27 + Q*Q/4

	if discriminant >= 0: # Complex numbers
		real = -Q/2 + discriminant**(1/2)
		imag = 0
	else:
		real = -Q/2
		imag = (-discriminant)**(1/2)

	norm_b3  = (real**2 + imag**2)**(1/2)
	phase_b3 = math.atan2(imag, real)

	norm_beta    = norm_b3**(1/3)
	phase_beta_1 = phase_b3/3
	phase_beta_2 = phase_beta_1 + 2/3*np.pi
	phase_beta_3 = phase_beta_2 + 2/3*np.pi

	if P >= 0: 
		norm_alpha = P/(3*norm_beta)
		phase_alpha_1 = -phase_beta_1
	else:
		norm_alpha = -P/(3*norm_beta)
		phase_alpha_1 = - phase_beta_1 + np.pi

	phase_alpha_2 = phase_alpha_1 + 4/3*np.pi
	phase_alpha_3 = phase_alpha_1 + 2/3*np.pi

	real_alpha_1 = norm_alpha*math.cos(phase_alpha_1)
	real_alpha_2 = norm_alpha*math.cos(phase_alpha_2)
	real_alpha_3 = norm_alpha*math.cos(phase_alpha_3)
	imag_alpha_1 = norm_alpha*math.sin(phase_alpha_1)
	imag_alpha_2 = norm_alpha*math.sin(phase_alpha_2)
	imag_alpha_3 = norm_alpha*math.sin(phase_alpha_3)
	real_beta_1  = norm_beta *math.cos(phase_beta_1 )
	real_beta_2  = norm_beta *math.cos(phase_beta_2 )
	real_beta_3  = norm_beta *math.cos(phase_beta_3 )
	imag_beta_1  = norm_beta *math.sin(phase_beta_1 )
	imag_beta_2  = norm_beta *math.sin(phase_beta_2 )
	imag_beta_3  = norm_beta *math.sin(phase_beta_3 )

	real_z_1 = real_alpha_1 - real_beta_1
	imag_z_1 = imag_alpha_1 - imag_beta_1
	real_z_2 = real_alpha_2 - real_beta_2
	imag_z_2 = imag_alpha_2 - imag_beta_2
	real_z_3 = real_alpha_3 - real_beta_3
	imag_z_3 = imag_alpha_3 - imag_beta_3

	t_1 = real_z_1 - a2/3
	t_2 = real_z_2 - a2/3
	t_3 = real_z_3 - a2/3

	if t0 < t_1 < t3 and imag_z_1 < 10**(-14):
		t = t_1
	elif t0 < t_2 < t3 and imag_z_2 < 10**(-14):
		t = t_2
	elif t0 < t_3 < t3 and imag_z_3 < 10**(-14):
		t = t_3
	else:
		print("Error: No solution in the correct range. Review the method")


	C = (t2-t)/(t2-t1)*((t2-t)/(t2-t0)*((t1-t)/(t1-t0)*P0   + \
		(t-t0)/(t1-t0)*P1)  + \
		(t-t0)/(t2-t0)*((t2-t)/(t2-t1)*P1   + \
		(t-t1)/(t2-t1)*P2)) + \
		(t-t1)/(t2-t1)*((t3-t)/(t3-t1)*((t2-t)/(t2-t1)*P1   + \
		(t-t1)/(t2-t1)*P2)  + \
		(t-t1)/(t3-t1)*((t3-t)/(t3-t2)*P2   + \
		(t-t2)/(t3-t2)*P3))

	return C

def Jet_coordinates(Point, jet_position, jet_width):
	'''
	Calculate Catmull-Rom for the chain of points and return the points of
	the jet
	'''
	# Convert the points to numpy so that one can do array multiplication
	P0, P1, P2, P3 = map(np.array, [Point[0], Point[1], Point[2], Point[3]])
	# Parametric constant: 0.5 for the centripetal spline, 0.0 for the uniform spline, 1.0 for the chordal spline.
	alpha = 0.5
	# Premultiplied power constant for the following tj() function.
	alpha = alpha/2

	t0 = 0
	t1 = tj(t0, P0, P1, alpha)
	t2 = tj(t1, P1, P2, alpha)
	t3 = tj(t2, P2, P3, alpha)

	C_l = solve_problem(P0, P1, P2, P3, t0, t1, t2, t3, jet_position - jet_width/2)
	C_r = solve_problem(P0, P1, P2, P3, t0, t1, t2, t3, jet_position + jet_width/2)

	return C_l, C_r

###########################################################################

case            = 'airfoil'
simu_name       = '2DAir'
dimension       = '2D' # Available : 2D, 3D
reward_function = 'lift_vs_drag' # at this moment, the only one used and parametrized

length         = 15
width          = 16
aoa            = 15
rotate_airfoil = 1  # 1: rotate airfoil. 0: rotate inlet velocity

sizeJet  = 0.004
sizeNaca = 0.008
sizeWake = 0.1
sizeFar  = 0.4

nb_proc     = 1 # Number of calculation processors
num_servers = 4 # number of environment in parallel

# Simulation params
simulation_duration   = 30 # to converge with velocity max = 1    
simulation_time_start = 0.0

delta_t_smooth   = 0.2      # smooth law duration
delta_t_converge = 0.0      # Total time that the DRL waits before executing a new action
smooth_func      = 'linear' # 'linear', 'parabolic', 'cubic' (cubic is still not coded)

# fluid properties
mu  = 2E-3
rho = 1.0

Dict_domain ={
	"downleft":  [-5,-width/2],
	"downright": [10,-width/2],
	"upright":   [10,width/2],
	"upleft":    [-5,width/2],
}

inlet_velocity = 1.5  # TODO: This is not used. It may can be deleted


## DRL Section
num_episodes = 400 # Total number of episodes

# Add jets. Each element in the vector represents 1 jet
jet_side     = [1, 1, 1]          # 1: jet on the suction side. -1: jet on the pressure side
jet_position = [0.4, 0.3, 0.2]    # Position of each jet along the chord
jet_width    = [0.02, 0.02, 0.02] # Width of each jet
n_jets       = len(jet_side)


## Get the position of the jets inside the airfoil profile
# Read airfoil profile
airfoil_file   = os.path.join(ALYA_PATH,'case','mesh','airfoil.dat')
airfoil_points = np.genfromtxt(airfoil_file) 

leading_edge_point = np.argmin(airfoil_points[:,0]) # Minimum of x coordinate

jet_coordinate_x1 = [] # To be deleted in favour of jets_definition
jet_coordinate_y1 = [] # To be deleted in favour of jets_definition
jet_coordinate_x2 = [] # To be deleted in favour of jets_definition
jet_coordinate_y2 = [] # To be deleted in favour of jets_definition
jet_name          = [] # To be deleted in favour of jets_definition

jets_definition = {}

for i in range(n_jets):
	name = "jet_{}".format(i+1)
	jet_name.append(name)
	
	if jet_side[i] == 1:  
		# Jet is in the suction side
		# Find the closest point of the airfoil to the left of the center
		# of the jet
		P3_id = np.argmax(airfoil_points[:,0] < jet_position[i])

		# Build segment
		segment = np.zeros((4,2),np.double)
		segment[:,0] = airfoil_points[P3_id-2:P3_id+2,0]
		segment[:,1] = airfoil_points[P3_id-2:P3_id+2,1]

	elif jet_side[i] == -1:  # Jet is in the pressure side
		# Find the closest point of the airfoil to the right of the center
		# of the jet
		P3_id = np.argmax(airfoil_points[leading_edge_point:-1,0] > jet_position[i])

		# Build segment
		segment = np.zeros((4,2),np.double)
		segment[:,0] = airfoil_points[leading_edge_point+P3_id-2:leading_edge_point+P3_id+2,0]
		segment[:,1] = airfoil_points[leading_edge_point+P3_id-2:leading_edge_point+P3_id+2,1]

	else:
		raise NotImplementedError('Jet side not implemented!')

	# Calculate the Catmull-Rom splines of the segment
	c_l, c_r = Jet_coordinates(segment,jet_position[i],jet_width[i])
	jets_definition[name] = {
		'width':  jet_width[i],
		'side' :  jet_side[i],
		'x1':     c_r[0],
		'y1':     c_r[1],
		'x2':     c_l[0],
		'y2':     c_l[1],
		'remesh': False
	}
	jet_coordinate_x1.append(c_r[0]) # To be deleted in favour of jets_definition
	jet_coordinate_y1.append(c_r[1]) # To be deleted in favour of jets_definition
	jet_coordinate_x2.append(c_l[0]) # To be deleted in favour of jets_definition
	jet_coordinate_y2.append(c_l[1]) # To be deleted in favour of jets_definition

# Build the jets
jets = build_jets(JetsAirfoil,jets_definition,delta_t_smooth)

geometry_params = { # Kept for legacy purposes but to be deleted when reworking the mesh script
	'output':           '.'.join(["airfoil", 'geo']),
	'jet_width':        jet_width,
	'jet_name' :        jet_name,
	'jet_side' :        jet_side,
	'jet_positions_x1': jet_coordinate_x1,
	'jet_positions_y1': jet_coordinate_y1,
	'jet_positions_x2': jet_coordinate_x2,
	'jet_positions_y2': jet_coordinate_y2,
	'remesh':           False
}

# Environment
# nb_actuations = 80 # Number of actuation of the neural network for each episode
nb_actuations = 20;

# PROBES POSITIONS
list_position_probes = []

positions_probes_for_grid_x = np.arange(0, 1.1, 0.1)
positions_probes_for_grid_y = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]

for crrt_x in positions_probes_for_grid_x:
	for crrt_y in positions_probes_for_grid_y:
		list_position_probes.append(np.array([crrt_x, crrt_y]))

positions_probes_for_grid_x = [1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5]
positions_probes_for_grid_y = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]

for crrt_x in positions_probes_for_grid_x:
	for crrt_y in positions_probes_for_grid_y:
		list_position_probes.append(np.array([crrt_x, crrt_y]))

len_airfoil_position_probes = 88
len_wake_position_probes    = 90

wake_range_min = len_airfoil_position_probes + 1
wake_range_max = len_airfoil_position_probes + len_wake_position_probes

tag_probs = {
	'airfoil': [1,len_airfoil_position_probes],
	'wake':    [wake_range_min,wake_range_max]
}

output_params = {
	'locations':  list_position_probes,
	'tag_probs':  tag_probs,
	'probe_type': 'pressure'
}

# post options
post_process_steps = 200

simulation_params = {
	'simulation_duration':  simulation_duration,#the time the simulation is in permanant regime
	'simulation_timeframe': [simulation_time_start,simulation_time_start+simulation_duration],
	'delta_t_smooth':       delta_t_smooth,
	'delta_t_converge':     delta_t_converge,
	'smooth_func':          smooth_func,
	'mu':                   mu,
	'rho':                  rho,
	'post_process_steps' :  post_process_steps
}

# Variational input 
variational_input = {
	'filename':        'airfoil', # basename 
	'bound':           [5],       # Boundaries to postprocess. Comma separeted
	'porous':          False,     # Variational boundaries to postprocess. Comma separeted 

	'density':         rho,       # Fluid density
	'veloc':           1,         # average velocity of a parabolic inflow
	"scale_area":      1,         # Projected frontal area. Scale if it is need it
	"d":               0,         # Distance for momentum calculation
	"time":            -0.158,    # Time to average. Negative to average from last time, 0 to average over total time, positive to postprocess form initial time
	"initial_time":    None,      # initial_time required only if positive "time averaging" used

	"rotation_vector": [1,0,0],   # Vector rotation in case rotation between axis is needed
	"phi":             -90,       # Rotation angle
	"D_exp":           3.23,      # SCHAFER Experimental Drag
	"S_exp":           0,         # Experimental Side
	"L_exp":           1.01,      # SCHAFER Experimental Lift
	"R_exp":           0,         # Experimental Roll
	"P_exp":           0,         # Experimental Pitch
	"Y_exp":           0          # Experimental yaw
}

# Optimization
optimization_params = {
	"num_steps_in_pressure_history": 1,
	"min_value_jet_MFR":             -1,
	"max_value_jet_MFR":             1,
	"norm_Q":                        0.01, # asa said in papers, limited Q for no momentum or discontinuities in the CFD solver
	"norm_reward":                   1, # like PRESS, try to be between -1,1
	"norm_press":                    0.4,  # force the PRESS comms to be between -1,1 (seggestion for better performance DRL)
	"offset_reward":                 -2.53, # start the reward from 0
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