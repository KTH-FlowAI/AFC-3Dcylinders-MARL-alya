#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Pol Suarez, Arnau Miro, Francisco Alcantara
# 07/07/2022
from __future__ import print_function, division

import os, subprocess

from configuration import ALYA_GMSH, ALYA_INCON
from env_utils     import run_subprocess


def run_mesh(runpath,casename,ndim,ini_vel=['0','0','0']):
	'''
	Use pyAlya tools to generate the mesh from gmsh
	'''
	# Build arguments string
	# Convert from GMSH to ALYA
	args  = '-2 ' if ndim == 2 else '' 
	args += '-c %s %s' % (casename,casename)
	run_subprocess(os.path.join(runpath,'mesh'),ALYA_GMSH,args)
	# TODO: generate periodicity if applicable
	# Generate initial condition
	args = '--vx %s --vy %s ' % (ini_vel[0],ini_vel[1])
	if len(ini_vel) > 2: args += '--vz %s ' % ini_vel[2]
	args += '%s' % casename
	run_subprocess(os.path.join(runpath,'mesh'),ALYA_INCON,args)
	# Symbolic link the mesh to the case main folder
	run_subprocess(runpath,'ln','-s mesh/*.mpio.bin .')


### Functions to write ALYA configuration files ###

def write_case_file(filepath,casename,simu_name):
	'''
	Writes the casename.dat file
	'''
	file = open(os.path.join(filepath,'%s.dat'%casename),'w')
	file.write('''$-------------------------------------------------------------------
RUN_DATA
  ALYA:                   %s
  INCLUDE                 run_type.dat
  LATEX_INFO_FILE:        YES
  LIVE_INFORMATION:       Screen
  TIMING:                 ON
END_RUN_DATA
$-------------------------------------------------------------------
PROBLEM_DATA
  TIME_COUPLING:          Global, from_critical
    INCLUDE               time_interval.dat
    NUMBER_OF_STEPS:      999999

  NASTIN_MODULE:          On
  END_NASTIN_MODULE

  PARALL_SERVICE          ON
    PARTITION_TYPE:       FACES
    POSTPROCESS:          MASTER
    PARTITIONING
    METHOD:               SFC
    EXECUTION_MODE:       PARALLEL
    END_PARTITIONING
  END_PARALL_SERVICE
END_PROBLEM_DATA
$-------------------------------------------------------------------
MPI_IO:        ON
  GEOMETRY:    ON
  RESTART:     ON
  POSTPROCESS: ON
END_MPI_IO
$-------------------------------------------------------------------'''%simu_name)
	file.close()

def write_run_type(filepath,type,freq=1):
	'''
	Writes the run type file that is included in the .dat
	'''
	file = open(os.path.join(filepath,'run_type.dat'),'w')
	# Write file
	file.write('RUN_TYPE: %s, PRELIMINARY, FREQUENCY=%d\n'%(type,freq))
	file.close()

def write_time_interval(filepath,start_time,end_time):
	'''
	Writes the time interval file that is included in the .dat
	'''
	file = open(os.path.join(filepath,'time_interval.dat'),'w')
	# Write file
	file.write('TIME_INTERVAL: %f, %f\n'%(start_time,end_time))
	file.close()

def detect_last_timeinterval(filename):
    # Open the file in read mode
    with open(filename, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

        # Loop through each line
        for line in lines:
            # Check if the line contains 'TIME_INTERVAL'
            if 'TIME_INTERVAL:' in line:
                # Find the position of 'TIME_INTERVAL:'
                time_interval_pos = line.find('TIME_INTERVAL:')
                # Extract the substring after 'TIME_INTERVAL:'
                substring = line[time_interval_pos + len('TIME_INTERVAL:'):]

                # Split the substring by comma to get individual values
                values = substring.split(',')

                # If there are values after 'TIME_INTERVAL:', return the first one
                if len(values) > 1:
                    return float(values[1].strip())

    # If no values are found after 'TIME_INTERVAL:', return None
    return None

def write_dom_file(filepath,casename,ncpus):
	'''
	Write the case_name.dom.dat
	'''
	file = open(os.path.join(filepath,'%s.dom.dat'%casename),'w')
	file.write('''$------------------------------------------------------------
DIMENSIONS
  INCLUDE	                  mesh/%s.dims.dat
  INCLUDE                   fields.dat
END_DIMENSIONS
$------------------------------------------------------------
STRATEGY
  GROUPS = %d, SEQUENTIAL_FRONTAL
  DOMAIN_INTEGRATION_POINTS:  0
  INTEGRATION_RULE:           OPEN
  BOUNDARY_ELEMENT:           OFF
  EXTRAPOLATE:                ON
  PERIODICITY:                MATRIX
END_STRATEGY
$-------------------------------------------------------------
GEOMETRY
END_GEOMETRY
$-------------------------------------------------------------
SETS
END_SETS
$-------------------------------------------------------------
BOUNDARY_CONDITIONS
END_BOUNDARY_CONDITIONS
$-------------------------------------------------------------
FIELDS
END_FIELDS
$-------------------------------------------------------------'''%(casename,ncpus))
	file.close()

def write_ker_file(filepath,casename,jetlist,steps,postprocess=[]):
	'''
	Write the casename.ker.dat

	postprocess can include CODNO, MASSM, COMMU, EXNOR
	'''
	# Create jet includes
	jet_includes = ''
	for jet in jetlist: jet_includes += '    INCLUDE %s.dat\n' % jet
	# Create variable postprocess
	var_includes = ''
	for var in postprocess: var_includes += '  POSTPROCESS %s\n' % var
	# Write file
	file = open(os.path.join(filepath,'%s.ker.dat'%casename),'w')
	file.write('''$------------------------------------------------------------
PHYSICAL_PROBLEM
  PROPERTIES
    INCLUDE       physical_properties.dat
  END_PROPERTIES
END_PHYSICAL_PROBLEM  
$------------------------------------------------------------
NUMERICAL_TREATMENT 
  MESH
    MASS:             CONSISTENT
    ELEMENTAL_TO_CSR: On
  END_MESH
  ELSEST
     STRATEGY: BIN
     NUMBER:   100,100,100
     DATAF:   LINKED_LIST
  END_ELSEST
  HESSIAN      OFF
  LAPLACIAN    OFF

  SPACE_&_TIME_FUNCTIONS
    INCLUDE inflow.dat
%s
  END_SPACE_&_TIME_FUNCTIONS
END_NUMERICAL_TREATMENT  
$------------------------------------------------------------
OUTPUT_&_POST_PROCESS 
  $ Variable postprocess 
  STEPS=%d
%s
  $ Witness points
  INCLUDE witness.dat
END_OUTPUT_&_POST_PROCESS  
$------------------------------------------------------------'''%(jet_includes,steps,var_includes))
	file.close()

def write_physical_properties(filepath,rho,mu):
	'''
	Writes the physical properties file that is included in the .ker.dat
	'''
	file = open(os.path.join(filepath,'physical_properties.dat'),'w')
	# Write file
	file.write('MATERIAL: 1\n')
	file.write('  DENSITY:   CONSTANT, VALUE=%f\n'%rho)
	file.write('  VISCOSITY: CONSTANT: VALUE=%f\n'%mu)
	file.write('END_MATERIAL\n')
	file.close()

def write_inflow_file(filepath,functions):
	'''
	Writes the inflow file that is included in the .ker.dat
	'''
	file = open(os.path.join(filepath,'inflow.dat'),'w')
	# Write file
	file.write('FUNCTION=INFLOW, DIMENSION=%d\n'%len(functions))
	for f in functions:
		file.write('  %s\n'%f)
	file.write('END_FUNCTION\n')
	file.close()	

def write_jet_file(filepath,name,functions):
	'''
	Writes the inflow file that is included in the .ker.dat
	'''
	file = open(os.path.join(filepath,'%s.dat'%name),'w')
	# Write file
	file.write('FUNCTION=%s, DIMENSION=%d\n'%(name.upper(),len(functions)))
	for f in functions:
		file.write('  %s\n'%f)
	file.write('END_FUNCTION\n')
	file.close()

def write_witness_file(filepath,probes_positions):
	'''
	Writes the witness file that is included in the .ker.dat
	'''
	nprobes = len(probes_positions)
	ndim    = len(probes_positions[0])
	# Open file for writing
	file    = open(os.path.join(filepath,'witness.dat'),'w')
	# Write header
	file.write('WITNESS_POINTS, NUMBER=%d\n'%nprobes)
	# Write probes
	if ndim == 2:
		for pos in probes_positions:
			file.write('%f,%f\n'%(pos[0],pos[1]))
	else:
		for pos in probes_positions:
			file.write('%f,%f,%f\n'%(pos[0],pos[1],pos[2]))
	# Write end
	file.write('END_WITNESS_POINTS\n')
	file.close()

def write_nsi_file(filepath,casename,varlist=['VELOC','PRESS'],witlist=['VELOX','VELOY','VELOZ','PRESS']):
	'''
	Write the casename.nsi.dat

	postprocess can include VELOC, PRESS, etc.
	'''
	# Create variable postprocess
	var_includes = ''
	for var in varlist: var_includes += '  POSTPROCESS %s\n' % var
	# Create jet includes
	wit_includes = ''
	for var in witlist: wit_includes += '    %s\n' % var
	# Write file
	file = open(os.path.join(filepath,'%s.nsi.dat'%casename),'w')
	file.write('''$------------------------------------------------------------
PHYSICAL_PROBLEM
  PROBLEM_DEFINITION       
    TEMPORAL_DERIVATIVES:	On  
    CONVECTIVE_TERM:	    EMAC
    VISCOUS_TERM:	        LAPLACIAN
  END_PROBLEM_DEFINITION  

  PROPERTIES
  END_PROPERTIES  
END_PHYSICAL_PROBLEM  
$------------------------------------------------------------
NUMERICAL_TREATMENT 
  TIME_STEP:            EIGENVALUE
  ELEMENT_LENGTH:       Minimum
  STABILIZATION:        OFF
  TIME_INTEGRATION:     RUNGE, ORDER:2
  SAFETY_FACTOR:        1.0
  STEADY_STATE_TOLER:   1e-10
  ASSEMBLY:             GPU2
  VECTOR:               ON
  NORM_OF_CONVERGENCE:  LAGGED_ALGEBRAIC_RESIDUAL
  MAXIMUM_NUMBER_OF_IT:	1
  GRAD_DIV:             ON
  DIRICHLET:            MATRIX

  ALGORITHM: SEMI_IMPLICIT
  END_ALGORITHM

  MOMENTUM
    ALGEBRAIC_SOLVER     
      SOLVER: EXPLICIT, LUMPED
    END_ALGEBRAIC_SOLVER        
  END_MOMENTUM

  CONTINUITY 
     ALGEBRAIC_SOLVER
       SOLVER:         DEFLATED_CG, COARSE: SPARSE
       CONVERGENCE:    ITERA=1000, TOLER=1.0e-10, ADAPTATIVE, RATIO=1e-2
       OUTPUT:         CONVERGENCE
       PRECONDITIONER: LINELET, NEVER_CHANGE
     END_ALGEBRAIC_SOLVER        
  END_CONTINUITY

  VISCOUS
    ALGEBRAIC_SOLVER
       SOLVER:         CG, COARSE: SPARSE, KRYLOV:10
       CONVERGENCE:    ITERA=500, TOLER=1.0e-10, ADAPTIVE, RATIO=1e-3
       OUTPUT:         CONVERGENCE
       PRECONDITIONER: DIAGONAL
    END_ALGEBRAIC_SOLVER    
  END_VISCOUS
END_NUMERICAL_TREATMENT  
$------------------------------------------------------------
OUTPUT_&_POST_PROCESS
  START_POSTPROCES_AT STEP  = 0
  $ Variables
%s  
  $ Forces at boundaries
  BOUNDARY_SET
	  FORCE
  END_BOUNDARY_SET
  $ Variables at witness points
  WITNESS_POINTS
%s
  END_WITNESS
END_OUTPUT_&_POST_PROCESS  
$------------------------------------------------------------
BOUNDARY_CONDITIONS, NON_CONSTANT
  PARAMETERS
    INITIAL_CONDITIONS: VALUE_FUNCTION = 1 $ use field 1 for initial condition
    FIX_PRESSURE:       OFF
    VARIATION:          NON_CONSTANT
  END_PARAMETERS
  $ Boundary codes
  INCLUDE boundary_codes.dat
END_BOUNDARY_CONDITIONS  
$------------------------------------------------------------'''%(var_includes,wit_includes))
	file.close()
