#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:00:04 2022

@author: francisco
"""

from __future__ import print_function, division

import os, sys
import copy as cp
import time

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from env_utils     import run_subprocess, generate_node_list, read_node_list
from configuration import ALYA_ULTCL

from Env3D_MARL import Environment

from cr import cr_reset, cr_info, cr_report

import time

from parameters import bool_restart, nb_inv_per_CFD, sync_episodes, batch_size, nb_actuations, num_episodes, num_servers, nb_proc, simu_name, run_baseline, nz_Qs 

# Ensure the chronometer is reset
cr_reset()

## Run
initial_time = time.time()

# Generate the list of nodes
# TODO --- ADD NUM_CFD (MARL)
generate_node_list(num_servers=num_servers,num_cores_server=nb_proc) # TODO: check if this works in MN!
#TODO: Update to local nodelists with num_servers

nodelist = read_node_list()

if not bool_restart:
    runbin  = 'rm -rf'
    path    = 'alya_files'
    args    = 'environment*'
    run_subprocess(path,runbin,args)

run_subprocess('alya_files','rm -rf','environment*/1/EP_*/flags_MARL') # Ensure deleting old parameters

# Now launch a single execution using mpirun -np <NPROCS> -host <HOSTNAME>
#assert len(hostlist) == num_servers # O mes fancy eh

#IMPORTANT: this environment base is needed to do the baseline, the main one
#TO DO: avoid baseline and manage the n-1 baseline that .py do, the solution for now is not fancy
environment_base = Environment(simu_name = simu_name, continue_training=True, node=nodelist[0]) # Baseline

# LA DEFINICIÓN DE AGENTE SE CARGA DEL CHECKPOINT, HAY QUE TESTEAR QUE COJA BIEN EL ARCHIVO
agent = Agent.load(directory=os.path.join(os.getcwd(), 'saver_data'), format='checkpoint', environment=environment_base)

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

#here the array of environments is defined, will be n-1 nodes (the 1st one is MASTER) #TODO: assign more nodes to an environment
parallel_environments = [Environment(simu_name = simu_name, continue_training=True, ENV_ID=[i,0], host="environment{}".format(i+1), node=nodelist[i+1]) for i in range(num_servers)]

environments = [split(parallel_environments[i], i+1)[j] for i in range(num_servers) for j in range(nz_Qs)]

for env in environments:
    print('Verif : ID ', env.ENV_ID, env.host)

for e in environments: 
    e.start() 
    time.sleep(2) 

#start all environments at the same time
#TO DO: needs a toy case for the start class a 'light' baseline for everyone which is useless
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
cr_report('DRL_TRAINING_conti.csv')
