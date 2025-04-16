# -*- coding: utf-8 -*-
"""
multi- agent VERSION 18/04/2024 

AUTHORS ->  POL

"""

###-----------------------------------------------------------------------------
## Import section

## IMPORT PYTHON LIBRARIES
import os, csv, numpy as np
import shutil
import time

# IMPORT TENSORFLOW 
from tensorforce.environments import Environment

# IMPORT INTERNAL LIBRARIES
from configuration    import ALYA_BIN, ALYA_GMSH, ALYA_SETS, ALYA_CLEAN, OVERSUBSCRIBE, DEBUG
from parameters       import bool_restart, neighbor_state, actions_per_inv, nb_inv_per_CFD, nz_Qs, mem_per_srun, dimension, case, simulation_params, num_nodes_srun, reward_function, jets, optimization_params, output_params, nb_proc, nb_actuations, nb_actuations_deterministic
from env_utils        import run_subprocess, printDebug
from alya             import write_case_file, write_witness_file, write_physical_properties, write_time_interval, write_run_type, detect_last_timeinterval
from extract_forces   import compute_avg_lift_drag
from witness          import read_last_wit
from cr               import cr_start, cr_stop
#from wrapper3D        import Wrapper
import copy as cp

###-------------------------------------------------------------------------###
###-------------------------------------------------------------------------###

### Environment definition
class Environment(Environment):
    

    ###---------------------------------------------------------------------###
    ###---------------------------------------------------------------------###
    
    ## Initialization of the environment
    ## only one time in multienvironment
    def __init__(self, simu_name, number_steps_execution=1, continue_training=False, deterministic=False, ENV_ID = [-1,-1],  host = '', node=None, check_id=False):
             
        cr_start('ENV.init',0)
        
        self.simu_name    = simu_name
        self.case         = case
        self.ENV_ID       = ENV_ID
        self.host         = "enviroment%d" %self.ENV_ID[0]
        self.nodelist         = node
        #self.nodelist     = [n for n in node.split(',')]
        self.do_baseline  = True # This parameter was being overwritten so it is no point to have it optional
        self.action_count = 0
        self.check_id     = check_id
        self.dimension    = dimension
        
        self.number_steps_execution = number_steps_execution
        self.reward_function        = reward_function
        self.output_params          = output_params
        self.optimization_params    = optimization_params
        self.Jets                   = jets
        self.n_jets                 = len(jets)
        self.nz_Qs                  = nz_Qs
        self.actions_per_inv        = actions_per_inv   
        self.nb_inv_per_CFD         = nb_inv_per_CFD
        self.bound_inv              = 6+self.ENV_ID[1]
        self.neighbor_state         = neighbor_state

        self.bool_restart_prevEP = bool_restart

        self.probes_values_global   = []     

        self.simulation_timeframe = simulation_params["simulation_timeframe"]
        self.last_time            = round(self.simulation_timeframe[1],3)
        self.delta_t_smooth       = simulation_params["delta_t_smooth"]
        self.smooth_func          = simulation_params["smooth_func"]

        self.previous_action_global = np.zeros(self.nb_inv_per_CFD)
        self.action_global          = np.zeros(self.nb_inv_per_CFD)
        
        #postprocess values
        self.history_parameters = {}
        self.history_parameters["drag"] = []
        self.history_parameters["lift"] = []
        self.history_parameters["drag_GLOBAL"] = []
        self.history_parameters["lift_GLOBAL"] = []
        self.history_parameters["time"] = []
        self.history_parameters["episode_number"] = []
        name="output.csv"
        # if we start from other episode already done
        last_row = None
        if(os.path.exists("saved_models/"+name)):
            with open("saved_models/"+name, 'r') as f:
                for row in reversed(list(csv.reader(f, delimiter=";", lineterminator="\n"))):
                    last_row = row
                    break
        if(not last_row is None):
            self.episode_number = int(last_row[0])
            self.last_episode_number = int(last_row[0])
        else:
            self.last_episode_number = 0
            self.episode_number = 0
        
        self.episode_drags = np.array([])
        self.episode_lifts = np.array([])
        self.episode_drags_GLOBAL = np.array([])
        self.episode_lifts_GLOBAL = np.array([])
        
        self.continue_training = continue_training
        self.deterministic     = deterministic

        if self.deterministic: self.host = 'deterministic'

        #check if the actual environment has to run cfd or not 
        # quick way --> if the 2nd component of the ENVID[] is 1...
        
        # Call parent class constructor
        super().__init__()
        
        cr_stop('ENV.init',0)
  
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
   
    def start(self):
        cr_start('ENV.start',0)
        # Get the new avg drag and lift and SAVE 
        temp_id = '{}'.format(self.host) if self.continue_training == True or self.deterministic == True else ''
        
        if self.continue_training:
            average_drag, average_lift = 0., 0.
            average_drag_GLOBAL, average_lift_GLOBAL = 0., 0.
        else:
            #average_drag, average_lift = compute_avg_lift_drag(self.episode_number, cpuid=temp_id)
            average_drag, average_lift = compute_avg_lift_drag(self.episode_number, cpuid=temp_id, nb_inv=self.ENV_ID[1]) #NOTE: add invariant code! not the same BC
            average_drag_GLOBAL, average_lift_GLOBAL = compute_avg_lift_drag(self.episode_number, cpuid=temp_id, nb_inv = self.nb_inv_per_CFD, global_rew = True) #NOTE: add invariant code! not the same BC

        # Update history parameters
        self.history_parameters["drag"].extend([average_drag])
        self.history_parameters["lift"].extend([average_lift])
        self.history_parameters["drag_GLOBAL"].extend([average_drag_GLOBAL])
        self.history_parameters["lift_GLOBAL"].extend([average_lift_GLOBAL])
        self.history_parameters["time"].extend([self.last_time])        
        self.history_parameters["episode_number"].extend([self.episode_number])
        self.save_history_parameters(nb_actuations)
        print("Results : \n\tAverage drag : {}\n\tAverage lift : {}".format(average_drag,average_lift))
        
        self.action = np.zeros(self.actions_per_inv*2)
        
        self.check_id = True # check if the folder with cpuid number is created
        cr_stop('ENV.start',0)

    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    
    def clean(self,full=False):
        cr_start('ENV.clean',0)
        if full:
            # saved_models contains the .csv of all cd and cl agt the end of each episode
            if os.path.exists("saved_models"): run_subprocess('./','rm -rf','saved_models')
            # Best model at the end of each episode
            if os.path.exists("best_model"): run_subprocess('./','rm -rf','best_model')
        # si no hemos acabado el episodio, continuamos sumando actions
        self.action_count = 1
        cr_stop('ENV.clean',0)
         
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    
    def create_mesh(self):  # TODO: Flag para que no tenga que volver a hacer la malla
        cr_start('ENV.mesh',0)
        if self.do_baseline == True:
            if self.dimension == 2:
                 run_subprocess('gmsh','python3','geo_file_maker.py') # TODO: this should be a library and be called within this function
                 run_subprocess('alya_files/case/mesh',ALYA_GMSH,'-2 %s'%self.case)
            for jet in self.Jets.values(): jet.update_file('alya_files/case')       
            write_witness_file('alya_files/case',output_params["locations"])
            run_subprocess('alya_files/case',ALYA_CLEAN,'')
        cr_stop('ENV.mesh',0)
        
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------

    def run_baseline(self,clean=True):
        cr_start('ENV.run_baseline',0)
        # Do a full clean
        if clean: self.clean(True)
        # Create the mesh
        self.create_mesh()
        # Setup alya files
        run_subprocess('alya_files','cp -r','case baseline') # TODO: substitute for correct case   
        #run_subprocess('alya_files/baseline/mesh','mv','*mpio.bin ..')
        if self.dimension == 2: run_subprocess('alya_files/baseline','python3','initialCondition.py {0} 1. 0.'.format(self.case))
        # Run alya
        self.run(which='reset')
        cr_stop('ENV.run_baseline',0)
            
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
           
    def run(self, which):
        print("Simulation on : ", self.simulation_timeframe)
        logssets = os.path.join('logs','log_sets.log')
        if which == 'reset':
            # Baseline run
            if self.do_baseline == True: # necessary? better?
                printDebug("\n \n Alya has started the baseline run! (Env2D-->run-->reset)\n \n")
                filepath = os.path.join('alya_files','baseline')
                write_case_file(filepath,self.case,self.simu_name)
                write_run_type(filepath,'NONCONTI',freq=1000)
                write_time_interval(filepath,self.simulation_timeframe[0],self.simulation_timeframe[1])
                write_physical_properties(filepath,simulation_params['rho'],simulation_params['mu'])
                # Run Alya
                casepath = os.path.join('alya_files','baseline')
                logsrun  = os.path.join('logs','log_last_reset_run.log')
                # Run subprocess
                if self.dimension == 2 :
                     run_subprocess(casepath,'mkdir -p','logs') # Create logs folder 
                     run_subprocess(casepath,ALYA_BIN,'%s'%self.case,nprocs=nb_proc,oversubscribe=OVERSUBSCRIBE,nodelist=self.nodelist,log=logsrun)#,parallel=True)
                     run_subprocess(casepath,ALYA_SETS,'%s-boundary.nsi.set 3'%self.case,log=logssets) # TODO: Boundary hardcoded!!
                if self.dimension == 3 :
                     run_subprocess(casepath,'mkdir -p','logs',preprocess=True) # Create logs folder
                     run_subprocess(casepath,ALYA_BIN,'%s'%self.case,nprocs=nb_proc,mem_per_srun=mem_per_srun,num_nodes_srun=num_nodes_srun,host=self.nodelist,log=logsrun)
                     run_subprocess(casepath,ALYA_SETS,'%s-boundary.nsi.set 3'%self.case,log=logssets,preprocess=True)

            self.do_baseline = False # Baseline done, no need to redo it       

        elif which == 'execute':
            # Actions run
            printDebug("\n \n Alya has started executing an action! (Env3D-->run-->execute) \n \n")
            cr_start('ENV.run_actions',0)
            filepath = os.path.join('alya_files','%s'%self.host,'%s' % self.ENV_ID[1],'EP_%d'%self.episode_number)

            filepath_flag_sync = os.path.join('alya_files','%s'%self.host,'1','EP_%s'%self.episode_number,'flags_MARL')
            action_end_flag_path  = os.path.join(filepath_flag_sync,'action_end_flag_%d' %self.action_count )
            time.sleep(0.1)

            if self.ENV_ID[1] == 1:

                write_run_type(filepath,'CONTI',freq=1000)
                write_time_interval(filepath,self.simulation_timeframe[0],self.simulation_timeframe[1])
                casepath = os.path.join('alya_files','%s'%self.host,'%s' %self.ENV_ID[1],'EP_%d'%self.episode_number)
                logsrun  = os.path.join('logs','log_last_execute_run.log' if not DEBUG else 'log_execute_run_%d.log'%self.action_count)
                # Run subprocess
                if self.dimension == 2:
                     run_subprocess(casepath,'mkdir -p','logs') # Create logs folder
                     run_subprocess(casepath,ALYA_BIN,'%s'%self.case,nprocs=nb_proc,oversubscribe=OVERSUBSCRIBE,nodelist=self.nodelist,log=logsrun)#,parallel=True)
                     run_subprocess(casepath,ALYA_SETS,'%s-boundary.nsi.set 3'%self.case,log=logssets) # TODO: Boundary hardcoded!!
                if self.dimension == 3:
                     run_subprocess(casepath,'mkdir -p','logs',preprocess=True) # Create logs folder
                     run_subprocess(casepath,ALYA_BIN,'%s'%self.case,nprocs=nb_proc,mem_per_srun=mem_per_srun,num_nodes_srun=num_nodes_srun,host=self.nodelist,log=logsrun)
                     run_subprocess(casepath,ALYA_SETS,'%s-boundary.nsi.set 3'%self.case,log=logssets,preprocess=True)
                
                # CREATE A FILE THAT WORKS AS FLAG TO THE OTHERS ENVS
                run_subprocess(filepath_flag_sync,'mkdir ','action_end_flag_%d' %self.action_count) # Create dir? not so elegant I think

            else:
                count_wait = 1 
                if not self.deterministic: 
                    while(not os.path.exists(action_end_flag_path) or not os.path.isdir(action_end_flag_path)):
                        if count_wait % 1000 == 0: 
                            print("Inv: %s is waiting for the action #%s" %(self.ENV_ID, self.action_count))
                        time.sleep(0.05)
                        count_wait += 1

                time.sleep(1)
                print("Actions in %s are sync" %self.ENV_ID)
                    

        
            cr_stop('ENV.run_actions',0)

    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
   
    def save_history_parameters(self, nb_actuations, name="output.csv"):
        
        cr_start('ENV.save_cd_cl',0)
        
        # Save at the end of every episodes
        self.episode_drags = np.append(self.episode_drags, self.history_parameters["drag"])
        self.episode_lifts = np.append(self.episode_lifts, self.history_parameters["lift"])
        self.episode_drags_GLOBAL = np.append(self.episode_drags_GLOBAL, self.history_parameters["drag_GLOBAL"])
        self.episode_lifts_GLOBAL = np.append(self.episode_lifts_GLOBAL, self.history_parameters["lift_GLOBAL"])
        
        if self.action_count == nb_actuations or self.episode_number == 0:
            file = os.path.join('saved_models',name)

            print("Action : saving history parameters in %s"%file)
            self.last_episode_number = self.episode_number
            
            avg_drag = np.mean(self.history_parameters["drag"][-1:])
            avg_lift = np.mean(self.history_parameters["lift"][-1:])
            avg_drag_GLOBAL = np.mean(self.history_parameters["drag_GLOBAL"][-1:])
            avg_lift_GLOBAL = np.mean(self.history_parameters["lift_GLOBAL"][-1:])
            
            os.makedirs('saved_models',exist_ok=True)
            if not os.path.exists("saved_models/"+name):
                with open(file,"w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Episode", "AvgDrag", "AvgLift", "AvgDrag_GLOBAL", "AvgLift_GLOBAL" ])
                    spam_writer.writerow([self.last_episode_number, avg_drag, avg_lift, avg_drag_GLOBAL, avg_lift_GLOBAL])
            else:
                with open(file, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.last_episode_number, avg_drag, avg_lift, avg_drag_GLOBAL, avg_lift_GLOBAL])
            self.episode_drags = np.array([])
            self.episode_lifts = np.array([])
            self.episode_drags_GLOBAL = np.array([])
            self.episode_lifts_GLOBAL = np.array([])
            
            # Writes all the cl and cd in .csv
            # IS THIS NECESSARY? I THINK WE DO NOT USE THE BEST MODEL
            if os.path.exists(file):
                run_subprocess('./','cp -r','saved_models best_model')
            else:
                if(os.path.exists("saved_models/output.csv")):
                    if(not os.path.exists("best_model")):
                        shutil.copytree("saved_models", "best_model")
                    else:
                        best_file = os.path.join('best_model',name)
                        last_iter = np.genfromtxt(file,skip_header=1,delimiter=';')[-1,1]
                        best_iter = np.genfromtxt(best_file,skip_header=1,delimiter=';')[-1,1]
                        if float(best_iter) < float(last_iter):
                            print("best_model updated")
                            run_subprocess('./','rm -rf','best_model')
                            run_subprocess('./','cp -r','saved_models best_model')
            printDebug("\n \n Saving parameters, AVG DRAG & AVG LIFT, which are the input of the neural network! (Env2D-->execute-->save_history_parameters)\n \n")
            print("Done.")
        cr_stop('ENV.save_cd_cl',0)
        
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    
    
    def save_this_action(self):
        
        cr_start('ENV.save_action',0)
    
        print("Saving a new action : N°",self.action_count)
        
        name_a = "output_actions.csv"
        if(not os.path.exists("actions")):
            os.mkdir("actions")
        
        if(not os.path.exists("actions/{}".format(self.host))):
            os.mkdir("actions/{}".format(self.host))
        
        if(not os.path.exists("actions/{}/{}_{}".format(self.host,self.ENV_ID[0],self.ENV_ID[1]))):
            os.mkdir("actions/{}/{}_{}".format(self.host,self.ENV_ID[0],self.ENV_ID[1]))
        
        if(not os.path.exists("actions/{}/{}_{}/ep_{}/".format(self.host, self.ENV_ID[0], self.ENV_ID[1], self.episode_number))):
            os.mkdir("actions/{}/{}_{}/ep_{}/".format(self.host, self.ENV_ID[0],self.ENV_ID[1], self.episode_number))
        
        path_a = "actions/{}/{}_{}/ep_{}/".format(self.host,self.ENV_ID[0],self.ENV_ID[1],self.episode_number)
        
        action_line = "{}".format(self.action_count)
        for i in range(self.actions_per_inv):
            action_line = action_line + "; {}".format(self.action[i])
        
        if(not os.path.exists(path_a+name_a)):
            header_line = "Action"
            for i in range(self.actions_per_inv):
                header_line = header_line + "; Jet_{}".format(i+1)
                
            with open(path_a+name_a, "w") as csv_file:
                spam_writer=csv.writer(csv_file, lineterminator="\n")
                spam_writer.writerow([header_line])
                spam_writer.writerow([action_line])
        else:
            with open(path_a+name_a, "a") as csv_file:
                spam_writer=csv.writer(csv_file, lineterminator="\n")
                spam_writer.writerow([action_line])
 
        
        print("Done.")
        
        cr_stop('ENV.save_action',0)

            
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    
    
    def save_reward(self,reward):
        
        cr_start('ENV.save_reward',0)
    
        print("Saving a new reward: N°", reward)
        
        name_a = "output_rewards.csv"
        
        if(not os.path.exists("rewards")):
            os.mkdir("rewards")

        if(not os.path.exists("rewards/{}".format(self.host))):
            os.mkdir("rewards/{}".format(self.host))
            
        if(not os.path.exists("rewards/{}/{}_{}".format(self.host,self.ENV_ID[0],self.ENV_ID[1]))):
            os.mkdir("rewards/{}/{}_{}".format(self.host,self.ENV_ID[0],self.ENV_ID[1]))
            
        if(not os.path.exists("rewards/{}/{}_{}/ep_{}/".format(self.host, self.ENV_ID[0], self.ENV_ID[1], self.episode_number))):
            os.mkdir("rewards/{}/{}_{}/ep_{}/".format(self.host, self.ENV_ID[0], self.ENV_ID[1], self.episode_number))
            
        path_a = "rewards/{}/{}_{}/ep_{}/".format(self.host, self.ENV_ID[0], self.ENV_ID[1], self.episode_number)
        
        if(not os.path.exists(path_a+name_a)):
                with open(path_a+name_a, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Action", "Reward"])#, "AvgRecircArea"])
                    spam_writer.writerow([self.action_count, reward])
        else:
                with open(path_a+name_a, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.action_count, reward])
                
        
        print("Done.")
        
        cr_stop('ENV.save_reward',0)
        
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
        
    def save_final_reward(self,reward):
    
        print("Saving the last reward from episode {}: ".format(self.episode_number), reward)
     
        name_a = "output_final_rewards.csv"
        
        if(not os.path.exists("final_rewards")):
            os.mkdir("final_rewards")

        if(not os.path.exists("final_rewards/{}".format(self.host))):
            os.mkdir("final_rewards/{}".format(self.host))
            
        if(not os.path.exists("final_rewards/{}/{}_{}".format(self.host,self.ENV_ID[0],self.ENV_ID[1]))):
            os.mkdir("final_rewards/{}/{}_{}".format(self.host,self.ENV_ID[0],self.ENV_ID[1]))
            
        path_a = "final_rewards/{}/{}_{}/".format(self.host,self.ENV_ID[0],self.ENV_ID[1])
        
        if(not os.path.exists(path_a+name_a)):
            with open(path_a+name_a, "w") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["EPISODE", "REWARD"])#, "AvgRecircArea"])
                spam_writer.writerow([self.episode_number, reward])
        else:
            with open(path_a+name_a, "a") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([self.episode_number, reward])
 
        print("Done.")
    
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
        
    def save_comms_probes(self): #TODO: This function is not used. May be eliminated
    
        print("Saving probes inputs: N°", self.action_count)
     
        name_a = "output_probes_comms.csv"
        
        if(not os.path.exists("probes_comms")):
            os.mkdir("probes_comms")
            
        if(not os.path.exists("probes_comms/ep_{}/".format(self.episode_number))):
            os.mkdir("probes_comms/ep_{}/".format(self.episode_number))
            
        path_a = "probes_comms/ep_{}/".format(self.episode_number)
        
        if(not os.path.exists(path_a+name_a)):
                with open(path_a+name_a, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    array_acts = np.linspace(1, 24, dtype=int) 
                    spam_writer.writerow(["Action", array_acts])#, "AvgRecircArea"])
                    spam_writer.writerow([self.action_count, self.probes_values])
        else:
                with open(path_a+name_a, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.action_count, self.probes_values])
 
        print("Done.")

    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------        
       
    ### AQUI DEBEMOS ANULAR EL RECUPERAR EL BASELINE SI YA EXISTE EL QUE TOCA
    def recover_start(self):

        cr_start('ENV.recover_start',0)

        

        runpath = 'alya_files'

        # flag to sync the cp times... then the other pseudo can read properly witness
        # path example: /alya_files/environment1/1/EP_1/.
        filepath_flag_sync_cp = os.path.join(runpath,'%s'%self.host,'1','EP_%d'%self.episode_number)

        #lowcost mode --> CLEAN everytime olderfiles

        #TODO --> check the rm if we need to restart from the last episode! 
        # TODO --> bool_restart_prevEP HAS TO BE 80-20 but not in parameters

        if not DEBUG and self.episode_number>0:
            if not self.bool_restart_prevEP:
                runbin  = 'rm -rf'            
                runargs = os.path.join('%s'%self.host,'%s'%self.ENV_ID[1],'EP_*')
                #avoid checks in deterministic
                if self.deterministic == False: run_subprocess(runpath,runbin,runargs)

        if self.bool_restart_prevEP and self.episode_number>1:
            runbin  = 'mv'
            runargs = '%s %s' %(os.path.join('%s'%self.host,'%s'%self.ENV_ID[1],'EP_%d'%(self.episode_number-1)),os.path.join('%s'%self.host,'%s'%self.ENV_ID[1],'EP_%d'%self.episode_number))
        else:
            runbin  = 'cp -r'
            runargs = 'baseline %s' %os.path.join('%s'%self.host,'%s'%self.ENV_ID[1],'EP_%d'%self.episode_number)
        
        logs    = os.path.join('baseline','logs','log_restore_last_episode_%d.log'%self.episode_number)
        
        run_subprocess(runpath,runbin,runargs,log=logs)
        #there was some data racin --> to avoid clean copies of flags... when doing bool_restart
        #run_subprocess('alya_files','rm -rf','environment*/1/EP_*/flags_MARL')

        run_subprocess(filepath_flag_sync_cp,'rm -rf','flags_MARL')
        run_subprocess(filepath_flag_sync_cp,'mkdir','flags_MARL') # Create dir? not so elegant I think
        run_subprocess(os.path.join(filepath_flag_sync_cp,'flags_MARL'),'rm -rf','*')
        run_subprocess(os.path.join(filepath_flag_sync_cp,'flags_MARL'),'mkdir','action_end_flag_cp') # Create dir? not so elegant I think

        cr_stop('ENV.recover_start',0)
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    # create folder for each cpu id in parallel and folder per invariants inside
    def create_cpuID(self):
        runpath = 'alya_files'
        runbin  = 'mkdir -p'
        #if self.deterministic == False:
        runargs = '%s' %self.host
        runpath2 = 'alya_files/%s' %self.host
        run_subprocess(runpath,runbin,runargs)

        for inv_i in range(1,self.nz_Qs+1):
            runargs2 = '%s' %inv_i            
            run_subprocess(runpath2,runbin,runargs2)
            
        # Write the nodes running this environmment
        name = "nodes"
        if(not os.path.exists("alya_files/{}/{}/".format(self.host, self.ENV_ID[1])+name)):
            with open("alya_files/{}/{}/".format(self.host, self.ENV_ID[1])+name, "w") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["Nodes in this learning"])
                spam_writer.writerow(self.nodelist)
        else:
            with open("alya_files/{}/{}/".format(self.host,self.ENV_ID[1])+name, "a") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["Nodes in this learning"])
                spam_writer.writerow(self.nodelist)
        #else:
        #    runargs = 'deterministic'
        #    run_subprocess(runpath,runbin,runargs,check_return=False)
            
        print('Folder created for CPU ID: %s/%s' % (self.host,self.ENV_ID[1]))
   
             
    # Optional
    def close(self):
        super().close()

    
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    ## Default function required for the DRL

    def list_observation(self):

        if not self.neighbor_state:
            #TODO: filter this observation state to each invariant and its neighbours: 
            batch_size_probes = int(len(self.probes_values_global)/self.nb_inv_per_CFD)
            probes_values_2 = self.probes_values_global[((self.ENV_ID[1]-1)*batch_size_probes):(self.ENV_ID[1]*batch_size_probes)]
        
        else:
            #TODO: filter this observation state to each invariant and its neighbours: 
            batch_size_probes = int(len(self.probes_values_global)/self.nb_inv_per_CFD)
        
            if self.ENV_ID[1] == 1:
                probes_values_halo  = self.probes_values_global[((self.nb_inv_per_CFD-1)*batch_size_probes):(self.nb_inv_per_CFD*batch_size_probes)]
                probes_values       = self.probes_values_global[((self.ENV_ID[1]-1)*batch_size_probes):((self.ENV_ID[1]+1)*batch_size_probes)]
                probes_values_2     = np.concatenate((probes_values_halo, probes_values))
                #print("POOOOOOOL len() line656:", probes_values_2)
                print("POOOOOOOL len() line657:", len(probes_values_2))
                print("POOOOOOOOOOOOOOOOL ---> TYPE PROBES ", type(probes_values_2))

            elif self.ENV_ID[1] == self.nb_inv_per_CFD:
                probes_values      = self.probes_values_global[((self.ENV_ID[1]-2)*batch_size_probes):(self.ENV_ID[1]*batch_size_probes)]
                probes_values_halo = self.probes_values_global[0:batch_size_probes]
                probes_values_2    = np.concatenate((probes_values, probes_values_halo))
                #print("POOOOOOOL len() line664:", probes_values_2)
                print("POOOOOOOL len() line665:", len(probes_values_2))
                print("POOOOOOOOOOOOOOOOL ---> TYPE PROBES ", type(probes_values_2))

            else:
                probes_values_2    = self.probes_values_global[((self.ENV_ID[1]-2)*batch_size_probes):((self.ENV_ID[1]+1)*batch_size_probes)]
                #print("POOOOOOOL len() line669:", probes_values_2)
                print("POOOOOOOL len() line670:", len(probes_values_2))
                print("POOOOOOOOOOOOOOOOL ---> TYPE PROBES ", type(probes_values_2))

        return probes_values_2
    
    def states(self):

        if not self.neighbor_state: 
            state_size = int(len(self.output_params["locations"])/self.nb_inv_per_CFD)
        else: 
            # TODO: introduce neighbours in parameters!
            # NOW IS JUST 1 EACH SIDE 85*3
            state_size = int(len(self.output_params["locations"])/self.nb_inv_per_CFD)*(3)

        return dict(type='float',
                    shape=(state_size, )
                    )
            
   #-----------------------------------------------------------------------------------------------------
   #-----------------------------------------------------------------------------------------------------    
        
    def actions(self):
        
        """Action is a list of n_jets-1 capped values of Q"""
        """UPDATE --> now with multiple Q per jet slot --> use nz_Qs"""
        """UPDATE 2 --> NOW WITH MARL --> ACTIONS_PER_INV = 1"""
 
        return dict(type='float',
                    shape=(self.actions_per_inv), 
                           min_value=self.optimization_params["min_value_jet_MFR"],
                           max_value=self.optimization_params["max_value_jet_MFR"]
                    )
    
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    
    
    def reset(self):

        #trick due sync issue when detecting time interval
        if self.ENV_ID[1] != 1: time.sleep(5)

        """Reset state"""
        print("\n \n Reset to initalize each episode (copy baseline, clean action count...)! (Env2D-->reset)\n \n")
        # Create a folder for each environment
        print("POOOOL --> CHECK_ID = ", self.check_id)
        print("POOOOL --> ENV_ID   = ", self.ENV_ID[1])
        if self.check_id == True and self.ENV_ID[1] == 1:
            self.create_cpuID()
            self.check_id = False
        
        # Clean
        print("\n\nLocation: Reset")
        print("Action: start to set up the case, set the initial conditions and clean the action counter")
        self.clean(False)

        # Advance in episode
        self.episode_number += 1
        
        # Apply new time frame
        # TODO --> fix time interval detected in the time_interval.dat file
        #     it has to read and detect self.simulation_timeframe[1] auto
        # wuick fix--dirty---> data racing when detecting last time interval
        self.simulation_timeframe = simulation_params["simulation_timeframe"]
        t1 = self.simulation_timeframe[0]
        if self.bool_restart_prevEP and self.episode_number>1: 
            if self.ENV_ID[1] == 1:
                t2 = detect_last_timeinterval(os.path.join('alya_files','%s'%self.host,'1','EP_%d'%(self.episode_number-1),'time_interval.dat'))
                print('POOOOOOOL PATH:', os.path.join('alya_files','%s'%self.host,'1','EP_%d'%(self.episode_number-1),'time_interval.dat'))
            else:
                t2 = detect_last_timeinterval(os.path.join('alya_files','%s'%self.host,'1','EP_%d'%(self.episode_number),'time_interval.dat'))
        else:
            t2 = self.simulation_timeframe[1]
        self.simulation_timeframe = [t1,t2]
        print("The actual timeframe is between {} and {}: ".format(t1,t2))
              
               
        # Copy the baseline in the environment directory  
        
        if self.action_count == 1 and self.ENV_ID[1]==1 :
            self.recover_start()
        
        if self.ENV_ID[1] == 1: 
            write_time_interval(os.path.join('alya_files','%s'%self.host,'%s'%self.ENV_ID[1],'EP_%d'%self.episode_number),t1,t2)

        print("Actual episode: {}".format(self.episode_number))
        print("\n\Action: extract the probes")
        NWIT_TO_READ=1 # Read n timesteps from witness file from behind, last instant

        # TODO: READ THE WITNESS OF EACH PSEUDOENV!
        # cp witness.dat to avoid IO problems in disk?
        # filename     = os.path.join('alya_files','%s'%self.host,'%s'%self.ENV_ID[1],'EP_%d'%self.episode_number,'%s.nsi.wit'%self.case)
        filename     = os.path.join('alya_files','%s'%self.host, '1','EP_%d'%self.episode_number,'%s.nsi.wit'%self.case)
        filepath_flag_sync_cp = os.path.join('alya_files','%s'%self.host,'1','EP_%d'%self.episode_number,'flags_MARL')
        
        action_end_flag_cp_path  = os.path.join(filepath_flag_sync_cp,'action_end_flag_cp')
        
        print("POOOOOOOL -> self.deterministic = ", self.deterministic)

        if not self.deterministic:
            while(not os.path.exists(action_end_flag_cp_path)):
                time.sleep(0.5)

        # Read witness file from behind, last instant (FROM THE INVARIANT running [*,1])
        NWIT_TO_READ         = 1
        filename             = os.path.join('alya_files','%s'%self.host,'1','EP_%d'%self.episode_number,'%s.nsi.wit'%self.case)

        #read witness file and extract the entire array list
        self.probes_values_global = read_last_wit(filename,output_params["probe_type"], self.optimization_params["norm_press"],NWIT_TO_READ)

        #filter probes per jet (corresponding to the ENV.ID[])
        probes_values_2      = self.list_observation()

        return probes_values_2
        
    #-----------------------------------------------------------------------------------------------------
    def execute(self, actions):
        
        action = []
        for i in range(self.actions_per_inv):
            action.append(self.optimization_params["norm_Q"]*actions[i])
            action.append(-self.optimization_params["norm_Q"]*actions[i])  # This is to ensure 0 mass flow rate in the jets
        
        #for i in range(self.actions_per_inv, self.actions_per_inv*2):
            #action.append(-self.optimization_params["norm_Q"]*actions[i-self.actions_per_inv])
        

        self.previous_action = self.action #save the older one to smooth the change
        self.action = action #update the new to reach at the end of smooth

        # Write the action
        self.save_this_action()

        print('New flux computed for INV: %s  :\n\tQs : %s' %(self.ENV_ID, self.action))
        
        dir_name = os.path.join('alya_files','%s'%self.host,'1','EP_%s' %self.episode_number,'flags_MARL')
        run_subprocess(dir_name,'mkdir','%d_inv_action_%d_ready' %(self.ENV_ID[1],self.action_count))
     
        self.last_time = self.simulation_timeframe[1]
        t1 = round(self.last_time,3)
        t2 = t1 + self.delta_t_smooth
            
        self.simulation_timeframe = [t1,t2]

        if self.ENV_ID[1]==1:

            cr_start('ENV.actions_MASTER_thread1',0)

            # wait until all the action from the others pseudoenvs are available
            all_actions_ready = False

            while not all_actions_ready:
                # loop over directory names
                for i in range(1, self.nb_inv_per_CFD):
                    dir_name = os.path.join('alya_files','%s'%self.host,'1','EP_%s' %self.episode_number,'flags_MARL','%d_inv_action_%d_ready' %(i,self.action_count))
                    all_actions_ready = True
                    if not os.path.exists(dir_name):
                        all_actions_ready = False
                time.sleep(0.2)
            
            print("**************** ALL ACTIONS ARE READY TO UPDATE BC *****************")
            #run_subprocess(dir_name,'rm -rf ','*_inv_action_*')


            ## NOW READY TO MERGE ACTIONS: 
            # append and reading file
            # open the file for reading

            for i in range(self.nb_inv_per_CFD):
                path_action_file = "actions/{}/{}_{}/ep_{}/output_actions.csv".format(self.host,self.ENV_ID[0],i+1,self.episode_number)
                with open(path_action_file, 'r') as file:
                    # read the lines of the file into a list
                    lines = csv.reader(file, delimiter=';')
                    # skip the header row
                    next(lines)
                    # initialize a variable to store the last value
                    last_action = None
                    # read each row and extract the second value
                    for row in lines:
                        last_action = float(row[1].strip())

                    #print("POOOOOOOOOOL -> last action : ", last_action)
                    self.previous_action_global[i] = self.action_global[i]
                    self.action_global[i] = last_action

            write_time_interval(os.path.join('alya_files','%s'%self.host,'%s'%self.ENV_ID[1],'EP_%d'%self.episode_number),t1,t2)
        
            simu_path = os.path.join('alya_files','%s'%self.host,'%s'%self.ENV_ID[1],'EP_%d'%self.episode_number)

            if self.case == 'cylinder':

                for ijet, jet in enumerate(self.Jets.values()): # Only need to loop on the values, i.e., Jet class
                    # Q_pre, Q_new, time_start, select smoothing law of the action
                    #print("POOOOOOL jet.update : ",self.previous_action_global,self.action_global,self.simulation_timeframe[0],self.smooth_func)
                    jet.update(self.previous_action_global,self.action_global,self.simulation_timeframe[0],self.smooth_func)
                    # Update the jet profile alya file
                    jet.update_file(simu_path)

            elif self.case == 'airfoil':
                for ijet, jet in enumerate(self.Jets.values()): # Only need to loop on the values, i.e., Jet class
                    # Update jets for the given epoch
                    if self.smooth_func == 'parabolic':
                        self.slope_pre = jet.slope_new
                    else:
                        self.slope_pre = 0
                    
                    # Q_pre, Q_new, time_start
                    jet.update(self.previous_action[ijet],self.action[ijet],self.simulation_timeframe[0])
                    # Update the jet profile alya file
                    jet.update_file(simu_path)

            cr_stop('ENV.actions_MASTER_thread1',0)


        
        # Start an alya run
        t0 = time.time()
        print("\n\nLocation : Execute/SmoothControl\nAction: start a run of Alya")
        self.run(which = 'execute')
        print("Done. time elapsed : ", time.time() - t0)
        
        # Get the new avg drag and lift --> LOCAL
        average_drag, average_lift = compute_avg_lift_drag(self.episode_number, cpuid = self.host, nb_inv=self.ENV_ID[1])
        self.history_parameters["drag"].extend([average_drag])
        self.history_parameters["lift"].extend([average_lift])
        self.history_parameters["time"].extend([self.last_time])
        self.history_parameters["episode_number"].extend([self.episode_number])
        self.save_history_parameters(nb_actuations)

        # Get the new avg drag and lift --> GLOBAL
        average_drag_GLOBAL, average_lift_GLOBAL = compute_avg_lift_drag(self.episode_number, cpuid = self.host, nb_inv = self.nb_inv_per_CFD, global_rew = True)
        self.history_parameters["drag_GLOBAL"].extend([average_drag_GLOBAL])
        self.history_parameters["lift_GLOBAL"].extend([average_lift_GLOBAL])

        
        # Compute the reward
        reward = self.compute_reward()
        self.save_reward(reward)
        print('reward: {}'.format(reward))
        
        print("The actual action is {} of {}".format(self.action_count, nb_actuations))

        self.action_count += 1
        
        if self.deterministic == False and self.action_count <= nb_actuations:
            terminal = False  # Episode is not done for training
        elif self.deterministic == True and self.action_count <= nb_actuations_deterministic:
            terminal = False  # Episode is not done for deterministic
        else:
            terminal = True   # Episode is done
            
            # write the last rewards at each episode to see the improvement 
            self.save_final_reward(reward)
            
            print("Actual episode: {} is finished and saved".format(self.episode_number))
            print("Results : \n\tAverage drag : {}\n\tAverage lift : {}".format(average_drag,average_lift))
        
        print("\n\Action : extract the probes")

        
        # Read witness file from behind, last instant (FROM THE INVARIANT running [*,1])
        NWIT_TO_READ=1
        filename      = os.path.join('alya_files','%s'%self.host,'1','EP_%d'%self.episode_number,'%s.nsi.wit'%self.case)
        
        #read witness file and extract the entire array list
        self.probes_values_global = read_last_wit(filename,output_params["probe_type"], self.optimization_params["norm_press"],NWIT_TO_READ)
        
        #filter probes per jet (corresponding to the ENV.ID[])
        probes_values_2      = self.list_observation()
        
        return probes_values_2, terminal, reward
    
    #-----------------------------------------------------------------------------------------------------
   
    def compute_reward(self):
        # NOTE: reward should be computed over the whole number of iterations in each execute loop
        if self.reward_function == 'plain_drag':  # a bit dangerous, may be injecting some momentum
            values_drag_in_last_execute = self.history_parameters["drag"][-1:]
            return(np.mean(values_drag_in_last_execute) + 0.159)  # the 0.159 value is a proxy value corresponding to the mean drag when no control; may depend on the geometry
        
        elif self.reward_function == 'drag_plain_lift_2':  # a bit dangerous, may be injecting some momentum
            avg_drag = np.mean(self.history_parameters["drag"])
            avg_lift = np.mean(self.history_parameters["lift"])
            return - avg_drag - 0.2 * abs(avg_lift)   
        
        elif self.reward_function == 'drag':  # a bit dangerous, may be injecting some momentum
            return self.history_parameters["drag"][-1] + 0.159
       
        elif self.reward_function == 'drag_plain_lift':  # a bit dangerous, may be injecting some momentum
            ## get the last mean cd or cl value of the last Tk
            avg_drag_2  = np.mean(self.history_parameters["drag"][-1:])
            avg_lift_2  = np.mean(self.history_parameters["lift"][-1:])
            avg_drag_2_global  = np.mean(self.history_parameters["drag_GLOBAL"][-1:])
            avg_lift_2_global  = np.mean(self.history_parameters["lift_GLOBAL"][-1:])

            reward_local  = (- avg_drag_2 - self.optimization_params["penal_cl"] * abs(avg_lift_2) + self.optimization_params["offset_reward"])
            reward_global = (- avg_drag_2_global - self.optimization_params["penal_cl"] * abs(avg_lift_2_global) + self.optimization_params["offset_reward"])
            print("POOOOOOOOL ---> reward_local: ", reward_local)
            print("POOOOOOOOL ---> reward_global: ", reward_global)
            print("POOOOOOOOL ---> cd_local: ", avg_drag_2)
            print("POOOOOOOOL ---> cd_lift: ", avg_lift_2)
            print("POOOOOOOOL ---> cd_local: ", avg_drag_2_global)
            print("POOOOOOOOL ---> cd_lift: ", avg_lift_2_global)

            alpha_rew     = self.optimization_params["alpha_rew"]
            reward_total = self.optimization_params["norm_reward"]*((alpha_rew)*reward_local + (1 - alpha_rew)*reward_global)
            
            ## le añadimos el offset de 3.21 para partir de reward nula y que solo vaya a (+)
            return reward_total
        
        elif self.reward_function == 'max_plain_drag':  # a bit dangerous, may be injecting some momentum
            values_drag_in_last_execute = self.history_parameters["drag"][-1:]
            return - (np.mean(values_drag_in_last_execute) + 0.159)
        
        elif self.reward_function == 'drag_avg_abs_lift':  # a bit dangerous, may be injecting some momentum
            avg_abs_lift = np.absolute(self.history_parameters["lift"][-1:])
            avg_drag = self.history_parameters["drag"][-1:]
            return avg_drag + 0.159 - 0.2 * avg_abs_lift
        
        elif self.reward_function == 'lift_vs_drag':  # a bit dangerous, may be injecting some momentum
            ## get the last mean cd or cl value of the last Tk
            avg_lift = np.mean(self.history_parameters["lift"][-1:])
            avg_drag = np.mean(self.history_parameters["drag"][-1:])
            
            return self.optimization_params["norm_reward"]*(avg_lift/avg_drag + self.optimization_params["offset_reward"])
