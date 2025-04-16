#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Pol Suarez, Arnau Miro, Francisco Alcantara, Xavi Garcia 
# 01/02/2023
from __future__ import print_function, division

import os, subprocess
from configuration import NODELIST, USE_SLURM, DEBUG


def run_subprocess(runpath,runbin,runargs,parallel=False,log=None,check_return=True,host=None,**kwargs):
	'''
	Use python to call a terminal command
	'''
	# Auxilar function to build parallel command
	def _cmd_parallel(runbin,**kwargs):
		'''
		Build the command to run in parallel
		'''
		nprocs = kwargs.get('nprocs',1)
		mem_per_srun = kwargs.get('mem_per_srun',1)
		num_nodes_srun = kwargs.get('num_nodes_srun',1)
		slurm = kwargs.get('slurm', USE_SLURM)
		
		# Switch to run srun or mpirun
		arg_hosts = ''
		if slurm:
			# Using srun
			arg_nprocs = '--nodes=%d --ntasks=%d --overlap --mem=%s' %(num_nodes_srun, nprocs, mem_per_srun) if nprocs > 1 else ''
			#arg_nprocs = '--ntasks=%d --overlap' %(nprocs) if nprocs > 1 else ''
			arg_export_dardel = 'export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)'
			launcher   = '%s && srun' %arg_export_dardel
			#launcher   = 'srun'
			arg_ovsub  = '--oversubscribe' if kwargs.get('oversubscribe',False) else ''
			if host != 'localhost' and host is not None:
				arg_hosts += '--nodelist=%s' % host
				#arg_hosts += ''
		else:
			# Using mpirun
			hostlist   = kwargs.get('nodelist',[])
			arg_nprocs = '-np %d --use-hwthread-cpus' % nprocs
			arg_ovsub  = '--oversubscribe' if kwargs.get('oversubscribe',False) else ''
			if host != 'localhost' and host is not None:
				arg_hosts += '-host %s' % host
			launcher   = 'mpirun'
		
		# Return the command
		return '%s %s %s %s %s' % (launcher,arg_ovsub,arg_nprocs,arg_hosts,runbin)
	
	# Check for a parallel run
	nprocs = kwargs.get('nprocs',1)
	if nprocs > 1: parallel = True # Enforce parallel if nprocs > 1
	
	# Check for logs
	arg_log = '> %s 2>&1' % log if log is not None else ''
	
	# Build command to run
	cmd_bin = _cmd_parallel('%s %s' % (runbin,runargs),**kwargs) if parallel else '%s %s' % (runbin,runargs)
	cmd     = 'cd %s && %s %s' % (runpath,cmd_bin,arg_log) #TODO: DARDEL DEBUG ONGOING
	#print('POOOOOOOOOOOOOL --> cmd: %s' % cmd)
	
	# Execute run
	retval = subprocess.call(cmd,shell=True)
	
	# Check return
	if check_return and retval != 0: raise ValueError('Error running command <%s>!'%cmd)
	
	# Return value
	return retval


def detect_system(override=None):
	'''
	Test if we are in a cluster or on a local machine
	'''
	# Override detect system and manually select the system
	if override is not None: return override
	# Start assuming we are on the local machine
	out = 'LOCAL' 
	# 1. Test for SRUN, if so we are in a SLURM machine
	# and hence we should use SLURM to check the available nodes
	if (run_subprocess('./','which','srun',check_return=False) == 0): out = 'SLURM'
	# Return system value
	return out


def _slurm_generate_node_list(outfile,num_servers,num_cores_server,**kwargs):
	'''
	Generate the list of nodes using slurm.
		> num_servers:      number of parallel runs
		> num_cores_server: number of cores per server using CFD
		> num_nodes:        number of nodes of the allocation (obtained from slurm environment variable)
		> num_cores_node:   number of cores per node (obtained from slurm environment variable)
	
	num_nodes and num_cores_node refer to the server configuration meanwhile 
	num_servers (number of environments in parallel) and num_cores_server (processors per environment)
	refer to the DRL configuration.

	SLURM_JOB_NODELIST does not give the exact list of nodes as we would want
	'''
	#print("POOOOOL --> SLURM_NNODES: %s" %os.getenv('SLURM_NNODES'))
	#print("POOOOOL --> SLURM_JOB_CPUS_PER_NODE: %s" %os.getenv('SLURM_JOB_CPUS_PER_NODE'))
	
	num_nodes      = kwargs.get('num_nodes',      int(os.getenv('SLURM_NNODES')))
	num_cores_node = kwargs.get('num_cores_node', os.getenv('SLURM_JOB_CPUS_PER_NODE'))
	start = num_cores_node.find('(')
	num_cores_node = int(num_cores_node[:start])
	num_cores_node = 100
	print("POOOOOL --> SLURM_JOB_CPUS_PER_NODE: %s" %num_cores_node)

	# Query SLURM to print the nodes used for this job to a temporal file
	# read it and store it as a variable
	run_subprocess('./','scontrol','show hostnames',log='tmp.nodelist') 
	hostlist = read_node_list(file='tmp.nodelist')
	#run_subprocess('./','rm','tmp.nodelist') 
	
	# Perform a sanity check
	if len(hostlist) != num_nodes: raise ValueError('Inconsistent number of nodes <%d> and hostlist <%d>!'%(num_nodes,len(hostlist))) # Ensure that we have read the hostlist correctly
	if num_servers*num_cores_server > (num_nodes)*num_cores_node+1: raise ValueError('Inconsistent allocation and DRL settings!') # Always ensure we have enough nodes for the task
	
	# Write the proper hostlist
	file = open(outfile,'w')
	
	# Leave the first node for the DRL only
	file.write('%s\n'%hostlist[0])
	# Write the rest of the nodes according to the allocation
	iserver = 0 #to debug in just 1 node be careful --- = 0 
	total_cores = num_cores_server #count the total servers to fill an entire node

	for ii in range(num_servers):
		# At least we will use one node
		total_cores += num_cores_server
		file.write('%s'%hostlist[iserver])
		# Integer division
		# to put more enviroments per node 
		
		if total_cores > num_cores_node:
			iserver +=1
			total_cores = num_cores_server
		# allocate more than one node. if num_cores_server < num_cores_node
		# then it will never get inside this for loop
		for jj in range(num_cores_server//(num_cores_node+1)):
			file.write(',%s'%hostlist[iserver])
			iserver +=1
		# Finish and jump line
		file.write('\n')

def _localhost_generate_node_list(outfile,num_servers):
	'''
	Generate the list of nodes for a local run
	'''
	hostlist = 'localhost'
	for iserver in range(num_servers): hostlist += '\nlocalhost'
	# Basically write localhost as the list of nodes
	# Add n+1 nodes as required per the nodelist
	run_subprocess('./','echo','"%s"'%hostlist,log=outfile)

def generate_node_list(override=None,outfile=NODELIST,num_servers=1,num_cores_server=1):
	'''
	Detect the system and generate the node list
	'''
	system  = detect_system(override)
	if system == 'LOCAL': _localhost_generate_node_list(outfile,num_servers)
	if system == 'SLURM': _slurm_generate_node_list(outfile,num_servers,num_cores_server)


def read_node_list(file=NODELIST):
	'''
	Read the list of nodes
	'''
	fp = open(file,'r')
	nodelist = [h.strip() for h in fp.readlines()]
	fp.close()
	return nodelist


def printDebug(*args):
	'''
...
	'''
	if DEBUG: print(*args)
