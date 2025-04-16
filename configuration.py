#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Path to the binaries used for the DRL tool
#
# Pol Suarez, Arnau Miro, Xavi Garcia
# 01/02/2023
from __future__ import print_function, division

import os


## PATHS
DRL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__))) # Absolute path to the DRL directory
ALYA_PATH = os.path.join(DRL_PATH,'alya_files')                      # Absolute path to the alya_files folder
BIN_PATH  = os.path.join(DRL_PATH,'alya_files','bin')                # Absolute path to the binaries folder


## FILE NAMES
NODELIST   = 'nodelist'


## BINARIES
ALYA_BIN   = os.path.join(BIN_PATH,'alya_3D_longfunc.x')
ALYA_SETS  = os.path.join(BIN_PATH,'alya-sets')
ALYA_CLEAN = os.path.join(BIN_PATH,'alya-clean')
ALYA_ULTCL = os.path.join(BIN_PATH,'ultra_cleaner.sh')
ALYA_GMSH  = os.path.join(BIN_PATH,'pyalya_gmsh2alya')
ALYA_PERIO = os.path.join(BIN_PATH,'pyalya_periodic')
ALYA_INCON = os.path.join(BIN_PATH,'pyalya_initialcondition')


## OPTIONS
# to use dardel ---> true
USE_SLURM = True


# Oversubscribe over MPI, must be set to False always
# unless debugging on the local machine and needed by
# the system
OVERSUBSCRIBE = True


# Activate the debug mode, which means you get prints of every action 
# and every action saves a log, that you can find in (enviroment1-->EP_1-->logs)

#It is recommended to use the debug mode with 1 enviroment and 1 proc, 
#in order not to mix prints from different enviroments
DEBUG = False
