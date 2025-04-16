#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Parallel training launcher
#
# Pol Suarez, Francisco Alcantara
# 21/02/2022
from __future__ import print_function, division

import os, sys
import copy as cp
import time

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from env_utils     import run_subprocess, generate_node_list, read_node_list
from configuration import ALYA_ULTCL

# Run the cleaner
run_subprocess('./',ALYA_ULTCL,'',preprocess=True) 

# Set up which case to run
training_case = "re3900_cylinder_3D_MARL"  # cylinder_2D, airfoil_2D, cylinder_3D
run_subprocess('./','rm -f','parameters.py',preprocess=True) # Ensure deleting old parameters
run_subprocess('./','ln -s','parameters/parameters_{}.py parameters.py'.format(training_case),preprocess=True)
run_subprocess('alya_files','cp -r','case_{} case'.format(training_case),preprocess=True)

from Env3D_MARL import Environment
from parameters import nb_inv_per_CFD, sync_episodes, batch_size, nb_actuations, num_episodes, num_servers, nb_proc, simu_name, run_baseline, nz_Qs 
from cr import cr_reset, cr_info, cr_report
import time

# Ensure the chronometer is reset
cr_reset()

## Run
initial_time = time.time()

# Generate the list of nodes
# TODO --- ADD NUM_CFD (MARL)
generate_node_list(num_servers=num_servers,num_cores_server=nb_proc) # TODO: check if this works in MN!
#TODO: Update to local nodelists with num_servers

# Read the list of nodes
nodelist = read_node_list()

#IMPORTANT: this environment base is needed to do the baseline, the main one
environment_base = Environment(simu_name=simu_name, node=nodelist[0]) # Baseline

if run_baseline: 
    run_subprocess('alya_files','rm -rf','baseline') # Ensure deleting old parameters
    environment_base.run_baseline(True)

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=environment_base, max_episode_timesteps=nb_actuations,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=batch_size, learning_rate=1e-3, subsampling_fraction=0.2, multi_step=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, predict_terminal_values=True,
    # TODO: gae_lambda=0.97 doesn't currently exist
    # Critic
    ## TODO -- memory ? 
    baseline=network,
    baseline_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    #TODO -- change parallel interaction_> how many calls to NN ?
    parallel_interactions=num_servers*nb_inv_per_CFD,
    # TensorFlow etc
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'), frequency=1, max_checkpoints=1),#parallel_interactions=number_servers,
    summarizer=dict(
        directory='data/summaries',
        # list of labels, or 'all'
        summaries=['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
    )
)

def split(environment, np):  # called 1 time in PARALLEL_TRAINING.py
    # np:= number of the parallel environment. e.g. between [1,4] for 4 parallel CFDenvironments
    # (ni, nj):= env_ID[1]:= 'position'/'ID-card' of the 'pseudo-parallel' invariant environment (a tuple in 3d, in which we have a grid of actuators. A scalar in 2D, in which we have a line of actuators)
    # nb_inv_envs:= total number of 'pseudo-parallel' invariant environments. e.g. 10
    ''' input: one of the parallel environments (np); output: a list of nb_inv_envs invariant environments identical to np. Their ID card: (np, ni)'''
    list_inv_envs = []
    for j in range(nz_Qs):
        env = cp.copy(environment)
        env.ENV_ID = [np, (j+1)]
        env.host="environment{}".format(np)
        list_inv_envs.append(env)
    return list_inv_envs

### Here the array of environments is defined, will be n-1 host (the 1st one is MASTER) #TODO: assign more nodes to an environment:
print('Here is the nodelist: ', nodelist)

#here the array of environments is defined, will be n-1 host (the 1st one is MASTER) #TODO: assign more nodes to an environment
parallel_environments = [Environment(simu_name=simu_name, ENV_ID=[i,0], host="environment{}".format(i+1), node=nodelist[i+1]) for i in range(num_servers)]

environments = [split(parallel_environments[i], i+1)[j] for i in range(num_servers) for j in range(nz_Qs)]

for env in environments:
    print('Verif : ID ', env.ENV_ID, env.host)

time.sleep(1.0)

#environments = [Environment(simu_name=simu_name, ENV_ID=i, host="environment{}".format(i+1),node=nodelist[i+1]) for i in range(num_servers)]
for e in environments: 
    e.start() 
    time.sleep(2) 

#start all environments at the same time
#TODO: needs a toy case for the start class a 'light' baseline for everyone which is useless
runner = Runner(agent=agent, environments=environments, remote='multiprocessing')

#now start the episodes and sync_episodes is very useful to update the DANN efficiently
runner.run(num_episodes=num_episodes, sync_episodes=sync_episodes)
runner.close()

#saving all the model data in model-numpy format 
agent.save(directory=os.path.join(os.getcwd(),'model-numpy'), format='numpy', append='episodes')

agent.close()

end_time = time.time()

print("DRL simulation :\nStart at : {}.\nEnd at {}\nDone in : {}".format(initial_time,end_time,end_time-initial_time))

cr_info()
cr_report('DRL_TRAINING.csv')
