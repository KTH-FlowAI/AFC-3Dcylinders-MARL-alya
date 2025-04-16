#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Deterministic run launcher
#
# Pol Suarez, Francisco Alcantara
# 21/02/2022
import os, time, sys
import copy as cp

from tensorforce.agents import Agent
from tensorforce.execution import Runner 

from env_utils  import run_subprocess, generate_node_list, read_node_list

from coordinate_folders_deterministic import adjust_folders #TODO: what is this??

from cr import cr_report, cr_start, cr_stop

import numpy as np

cr_start('deterministic-100-3D',0)

# Run the cleaner
#run_subprocess('./',ALYA_ULTCL,'') 

# Set up which case to run
training_case = "cylinder_3D_MARL"  # cylinder_2D, airfoil_2D
run_subprocess('./','rm -f','parameters.py') # Ensure deleting old parameters
run_subprocess('./','cp','parameters/parameters_{}.py parameters.py'.format(training_case))
run_subprocess('alya_files','cp -r','case_{} case'.format(training_case))

runbin  = 'rm -rf'
path    = 'alya_files'
args    = 'environment*'
run_subprocess(path,runbin,args)

runbin  = 'rm -rf'
path    = './'
args    = 'best_model saved_models rewards final_rewards actions'
run_subprocess(path,runbin,args)

from Env3D_MARL      import Environment
from parameters import nb_inv_per_CFD, sync_episodes, batch_size, nb_actuations, nb_actuations_deterministic, num_episodes, num_servers, nb_proc, simu_name, run_baseline, nz_Qs, optimization_params,output_params, neighbor_state

from threading import Thread
from multiprocessing import Process

from witness          import read_last_wit


## Run
initial_time = time.time()

# Generate the list of nodes
# TODO: hardcoded for 1 node/env???
generate_node_list(num_servers=num_servers,num_cores_server=nb_proc) 

# Read the list of nodes
nodelist = read_node_list()

#TODO: assign more nodes to an environment
environment_base = Environment(simu_name = simu_name, node=nodelist[1], deterministic=True) 
#Check parameters to run_baseline flag
if run_baseline: environment_base.run_baseline(True)

# Load agent TensorFlow checkpoint
agent = Agent.load(directory=os.path.join(os.getcwd(), 'saver_data'), format='checkpoint', environment=environment_base)

def list_observation(probes_values_global, ENV_ID, nb_inv_per_CFD):

    if not neighbor_state:
        #TODO: filter this observation state to each invariant and its neighbours: 
        batch_size_probes = int(len(probes_values_global)/nb_inv_per_CFD)
        probes_values_2 = probes_values_global[((ENV_ID[1]-1)*batch_size_probes):(ENV_ID[1]*batch_size_probes)]
    
    else:
        #TODO: filter this observation state to each invariant and its neighbours: 
        batch_size_probes = int(len(probes_values_global)/nb_inv_per_CFD)
    
        if ENV_ID[1] == 1:
            probes_values_halo  = probes_values_global[((nb_inv_per_CFD-1)*batch_size_probes):(nb_inv_per_CFD*batch_size_probes)]
            probes_values       = probes_values_global[((ENV_ID[1]-1)*batch_size_probes):((ENV_ID[1]+1)*batch_size_probes)]
            probes_values_2     = np.concatenate((probes_values_halo, probes_values))

        elif ENV_ID[1] == nb_inv_per_CFD:
            probes_values      = probes_values_global[((ENV_ID[1]-2)*batch_size_probes):(ENV_ID[1]*batch_size_probes)]
            probes_values_halo = probes_values_global[0:batch_size_probes]
            probes_values_2    = np.concatenate((probes_values, probes_values_halo))

        else:
            probes_values_2    = probes_values_global[((ENV_ID[1]-2)*batch_size_probes):((ENV_ID[1]+1)*batch_size_probes)]

    return probes_values_2

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
parallel_environments = [Environment(simu_name = simu_name, deterministic=True, ENV_ID=[i,0], host="deterministic", node=nodelist[i+1]) for i in range(num_servers)]

environments = [split(parallel_environments[i], i+1)[j] for i in range(num_servers) for j in range(nz_Qs)]

for env in environments:
    print('Verif : ID ', env.ENV_ID, env.host)

time.sleep(1.0)

list_states = []

for e in environments:
    e.start()
    list_states.append(e.reset())
    time.sleep(0.2) 

# to run evaluation==True + multiprocessing/multienv 
for action_step in range(nb_actuations_deterministic):    
    for e in reversed(environments):
        #update action list --> merged   
        list_actions = [agent.act(list_states[e.ENV_ID[1]-1], deterministic=True, independent = True)]
        print('New action generated! ID ', env.ENV_ID, env.host, list_actions)
        #update state + update BC --> becareful data race (nightmare)
        list_states[e.ENV_ID[1]-1], terminal, reward = e.execute(list_actions[0])
        print('CFD + action finished! ID ', env.ENV_ID, env.host)

    #we need to rewrite manually the state of the non-CFD runners (execute was doing that for us but we need to split)
    for e in environments:
        if e.ENV_ID[1] != 1:            
            # Read witness file from behind, last instant (FROM THE INVARIANT running [*,1])
            nwit_to_read        = 1
            filename             = os.path.join('alya_files','environment1','1','EP_1','cylinder.nsi.wit')

            #read witness file and extract the entire array list
            probes_values_global = read_last_wit(filename,output_params["probe_type"], optimization_params["norm_press"], nwit_to_read)

            #filter probes per jet (corresponding to the ENV.ID[])
            list_states[e.ENV_ID[1]-1] = list_observation(nb_inv_per_CFD=nb_inv_per_CFD, probes_values_global=probes_values_global,ENV_ID=e.ENV_ID)
            print('non-CFD states updated! ID ', env.ENV_ID, env.host)

agent.close()

# Adjust folders for CONTINUE_LEARNING
adjust_folders(num_servers)

end_time = time.time()
cr_stop('deterministic-100-3D',0)
cr_report('deterministic.profile.csv')

print("DRL deterministic simulation :\nStart at : {}.\nEnd at {}\nDone in : {}".format(initial_time,end_time,end_time-initial_time))
