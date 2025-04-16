#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:00:04 2022

@author: francisco
"""
import os

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from Env3D import Environment

from cr import cr_reset, cr_info, cr_report

import time

from parameters import num_episodes, simu_name, num_servers

cr_reset()

## Run
initial_time = time.time()

# Read the list of nodes
fp = open('nodelist','r')
nodelist = [h.strip() for h in fp.readlines()]
fp.close()

# Now launch a single execution using mpirun -np <NPROCS> -host <HOSTNAME>
#assert len(hostlist) == num_servers # O mes fancy eh

#IMPORTANT: this environment base is needed to do the baseline, the main one
#TO DO: avoid baseline and manage the n-1 baseline that .py do, the solution for now is not fancy
environment_base = Environment(simu_name = simu_name, continue_training=True, ENV_ID=-1, node=nodelist[0]) # Baseline

# LA DEFINICIÓN DE AGENTE SE CARGA DEL CHECKPOINT, HAY QUE TESTEAR QUE COJA BIEN EL ARCHIVO
agent = Agent.load(directory=os.path.join(os.getcwd(), 'saver_data'), format='checkpoint', environment=environment_base)

#here the array of environments is defined, will be n-1 nodes (the 1st one is MASTER) #TODO: assign more nodes to an environment
environments = [Environment(simu_name = simu_name, continue_training=True, ENV_ID=i, host="environment{}".format(i+1), node=nodelist[i+1]) for i in range(num_servers)]

#start all environments at the same time
#TO DO: needs a toy case for the start class a 'light' baseline for everyone which is useless
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
