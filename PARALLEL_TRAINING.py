#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Parallel training launcher
#
# Pol Suarez, Francisco Alcantara
# 21/02/2022
from __future__ import print_function, division

import os

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from env_utils     import run_subprocess, generate_node_list, read_node_list
from configuration import ALYA_ULTCL

# Run the cleaner
run_subprocess('./',ALYA_ULTCL,'') 

# Set up which case to run
training_case = "cylinder_3D"  # cylinder_2D, airfoil_2D, cylinder_3D
run_subprocess('./','rm -f','parameters.py') # Ensure deleting old parameters
run_subprocess('./','ln -s','parameters/parameters_{}.py parameters.py'.format(training_case))
run_subprocess('alya_files','cp -r','case_{} case'.format(training_case))

from Env3D import Environment3D
from parameters import nb_actuations, num_episodes, num_servers, simu_name

from cr import cr_reset, cr_info, cr_report
import time

# Ensure the chronometer is reset
cr_reset()


## Run
initial_time = time.time()

# Generate the list of nodes
generate_node_list(num_servers=num_servers) # TODO: check if this works in MN!
#TODO: Update to local nodelists with num_servers

# Read the list of nodes
nodelist = read_node_list()

#IMPORTANT: this environment base is needed to do the baseline, the main one
environment_base = Environment3D(simu_name=simu_name,node=nodelist[0]) # Baseline

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=environment_base, max_episode_timesteps=nb_actuations,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=20, learning_rate=1e-3, subsampling_fraction=0.2, multi_step=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, predict_terminal_values=True,
    # TODO: gae_lambda=0.97 doesn't currently exist
    # Critic
    baseline=network,
    baseline_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    parallel_interactions=num_servers,
    # TensorFlow etc
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'), frequency=1, max_checkpoints=1),#parallel_interactions=number_servers,
)

#here the array of environments is defined, will be n-1 host (the 1st one is MASTER) #TODO: assign more nodes to an environment
environments = [Environment3D(simu_name=simu_name, do_baseline=False, ENV_ID=i, host="environment{}".format(i+1), node=nodelist[i+1]) for i in range(num_servers)]

#start all environments at the same time
#TODO: needs a toy case for the start class a 'light' baseline for everyone which is useless
runner = Runner(agent=agent, environments=environments, remote='multiprocessing')

#now start the episodes and sync_episodes is very useful to update the DANN efficiently
runner.run(num_episodes=num_episodes, sync_episodes=True)
runner.close()

#saving all the model data in model-numpy format 
agent.save(directory=os.path.join(os.getcwd(),'model-numpy'), format='numpy', append='episodes')

agent.close()

end_time = time.time()

print("DRL simulation :\nStart at : {}.\nEnd at {}\nDone in : {}".format(initial_time,end_time,end_time-initial_time))

cr_info()
cr_report('DRL_TRAINING.csv')
