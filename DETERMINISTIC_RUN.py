#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Deterministic run launcher
#
# Pol Suarez, Francisco Alcantara
# 21/02/2022
import os, time

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from env_utils  import run_subprocess, generate_node_list, read_node_list

from coordinate_folders_deterministic import adjust_folders #TODO: what is this??

from cr import cr_report, cr_start, cr_stop

cr_start('DRL-100-3D',0)

# Run the cleaner
#run_subprocess('./',ALYA_ULTCL,'') 

# Set up which case to run
training_case = "cylinder_3D"  # cylinder_2D, airfoil_2D
run_subprocess('./','rm -f','parameters.py') # Ensure deleting old parameters
run_subprocess('./','cp','parameters/parameters_{}.py parameters.py'.format(training_case))
run_subprocess('alya_files','cp -r','case_{} case'.format(training_case))

from Env3D_beta      import Environment
from parameters import simu_name, num_servers, nb_proc, run_baseline

## Run
initial_time = time.time()

# Generate the list of nodes
generate_node_list(num_servers=num_servers,num_cores_server=nb_proc) 

# Read the list of nodes
nodelist = read_node_list()

#TODO: assign more nodes to an environment
environment_base = Environment(simu_name = simu_name, node=nodelist[0], deterministic=True) 
#Check parameters to run_baseline flag
if run_baseline: environment_base.run_baseline(True)

# Load agent TensorFlow checkpoint
agent = Agent.load(directory=os.path.join(os.getcwd(), 'saver_data_re100_3D_it03'), format='checkpoint', environment=environment_base)

#TODO: check new nb_actuations to have bigger episodes
runner = Runner(agent=agent, environment=environment_base, max_episode_timesteps=5000)

runner.run(num_episodes=1, evaluation=True)
runner.close()
agent.close()

# Adjust folders for CONTINUE_LEARNING
adjust_folders(num_servers)

end_time = time.time()
cr_stop('DRL-100-3D',0)
cr_report('DRL.CONTINUE.csv')

print("DRL simulation :\nStart at : {}.\nEnd at {}\nDone in : {}".format(initial_time,end_time,end_time-initial_time))
