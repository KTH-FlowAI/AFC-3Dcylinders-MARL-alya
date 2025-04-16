# -*- coding: utf-8 -*-
"""
multi- ENVIRONMENT VERSION 1/6/2022 

AUTHORS -> FRAN, PAU, POL

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
from configuration    import ALYA_BIN, ALYA_GMSH, ALYA_SETS, ALYA_CLEAN
from parameters       import case, simulation_params, reward_function, jets, optimization_params, output_params, nb_proc, nb_actuations, nb_actuations_deterministic
from env_utils        import run_subprocess
from alya             import write_witness_file, write_physical_properties, write_time_interval, write_run_type
from extract_forces   import compute_avg_lift_drag
from witness          import read_last_wit
from cr               import cr_start, cr_stop


###-------------------------------------------------------------------------###
###-------------------------------------------------------------------------###

### Environment definition
class Environment2D(Environment):
    

    ###---------------------------------------------------------------------###
    ###---------------------------------------------------------------------###
    
    ## Initialization of the environment
    ## only one time in multienvironment
    def __init__(self, simu_name, number_steps_execution=1, do_baseline=True, continue_training=False, deterministic=False, ENV_ID = -1,  host = '', node=None):
             
        cr_start('ENV.init',0)
        
        self.simu_name = simu_name
        self.case      = case
        self.ENV_ID    = ENV_ID
        self.host      = host
        self.node      = node

        self.number_steps_execution = number_steps_execution
        self.reward_function        = reward_function
        self.output_params          = output_params
        self.optimization_params    = optimization_params
        self.Jets                   = jets
        self.n_jets                 = len(jets)
        
        self.simulation_timeframe = simulation_params["simulation_timeframe"]
        self.last_time            = round(self.simulation_timeframe[1],3)
        self.delta_t_smooth       = simulation_params["delta_t_smooth"]
        self.smooth_func          = simulation_params["smooth_func"]
        
        #postprocess values
        self.history_parameters = {}
        self.history_parameters["drag"] = []
        self.history_parameters["lift"] = []
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
        
        self.do_baseline = do_baseline
        self.continue_training = continue_training
        self.deterministic = deterministic
        
        self.start_class()
        
        self.do_baseline = True
        
        #start episodes
        super().__init__()
        
        cr_stop('ENV.init',0)
  
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
   
    ## Start baseline_flow from scratch, creating mesh and cfd setup
    
    def start_class(self):
        #----------------------------------------------------------------------
        # Set up and clean
        #t0 ->only computational cost
        t0 = time.time()
        
        print("\n\nLocation : Start_class\nSTATE : start to set up the case and clean")

        #remesh de todo y limpiamos backups
        self.clean(True)
        
        #ahora tenemos el case base como baseline y es donde hacemos run del flow estabilizado
        self.create_baseline()
        
        print("Done. Time elapsed : ", time.time() - t0,"\n\n")
        #----------------------------------------------------------------------
        # Run the first case
        t0 = time.time()
        print("\n\nLocation : Start_class\nSTATE : start a run of Alya")
        
        #alya run in baseline 
        self.run(which = 'reset')
       
        print("Done. time elapsed : ", time.time() - t0)
        
        # Get the new avg drag and lift and SAVE 
        self.action_count=0
        
        if self.continue_training == True or self.deterministic == True:
            temp_id = '{}'.format(self.host)
        else:
            temp_id = ''
        
        if self.continue_training:
            average_drag, average_lift = 0., 0.
        else:
            average_drag, average_lift = compute_avg_lift_drag(self.episode_number, cpuid=temp_id)
        self.history_parameters["drag"].extend([average_drag])
        self.history_parameters["lift"].extend([average_lift])
        self.history_parameters["time"].extend([self.last_time])        
        self.history_parameters["episode_number"].extend([self.episode_number])
        self.save_history_parameters(nb_actuations)
        
        print("Results : \n\tAverage drag : {}\n\tAverage lift : {}".format(average_drag,average_lift))
        
        #check if the folder with cpuid number is created
        self.check_id = True

    
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    
    def clean(self,full):
        
        cr_start('ENV.clean',0)
        
        if full:
            if self.do_baseline == True:
                ## saved_models contains the .csv of all cd and cl agt the end of each episode
                if os.path.exists("saved_models"):
                    run_subprocess('./','rm -r','saved_models')
                
                ## Best model at the end of each episode
                if os.path.exists("best_model"):
                    run_subprocess('./','rm -r','best_model')
            
            ## Generate mesh from *.geo
            
            ## UPDATE ## boundaries included
            self.create_mesh()
            
        else:
            ## si no hemos acabado el episodio, continuamos sumando actions
            self.action_count = 1
            
        cr_stop('ENV.clean',0)
         
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    
    def create_mesh(self):  # TODO: Flag para que no tenga que volver a hacer la malla
        cr_start('ENV.mesh',0)
        
        if self.do_baseline == True:
            run_subprocess('gmsh','python3','geo_file_maker.py') # TODO: this should be a library and be called within this function
            run_subprocess('alya_files/case/mesh',ALYA_GMSH,'-2 %s'%self.case)
            for jet in self.Jets.values(): jet.update_file('alya_files/case')       
            write_witness_file('alya_files/case',output_params["locations"])
            run_subprocess('alya_files/case',ALYA_CLEAN,'')
        
        cr_stop('ENV.mesh',0)
        
    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    
    def create_baseline(self):
        if self.do_baseline == True:
            run_subprocess('alya_files','cp -r','case baseline')      
            run_subprocess('alya_files/baseline/mesh','mv','*mpio.bin ..')
            run_subprocess('alya_files/baseline','python3','initialCondition.py {0} 1. 0.'.format(self.case))

    #-------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
           
    def run(self, which):
        
        print("Simulation on : ", self.simulation_timeframe)
        
        if which == 'reset':
            cr_start('ENV.run_baseline',0)
            
            if self.do_baseline == True:
                ## CONFIGURE *DAT alya
                filepath = os.path.join('alya_files','baseline')
                write_time_interval(filepath,self.simulation_timeframe[0],self.simulation_timeframe[1])
                write_physical_properties(filepath,simulation_params['rho'],simulation_params['mu'])
            
            #TODO: do not do a restart from the beginning if baseline already exists
            self.switch(state = 'restart')
            self.run_alya(which)
            self.switch(state = 'continue') # now begin episodes
            
            cr_stop('ENV.run_baseline',0)
            
        elif which == 'execute':
            cr_start('ENV.run_actions',0)
            
            filepath = os.path.join('alya_files','%s'%self.host,'EP_%d'%self.episode_number)
            write_time_interval(filepath,self.simulation_timeframe[0],self.simulation_timeframe[1])
            self.run_alya(which)
            
            cr_stop('ENV.run_actions',0)
    
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    
    def run_alya(self, which):
        
        if which == 'reset': ## only for the baseline_case
            if self.do_baseline == False:
                return
            
            casepath = os.path.join('alya_files','baseline')
            logsrun  = os.path.join('logs','log_last_reset_run.log')
            logssets = os.path.join('logs','log_sets.log')
            # Run subprocess
            run_subprocess(casepath,'mkdir -p','logs') # Create logs folder
            run_subprocess(casepath,ALYA_BIN,'%s'%self.case,nprocs=nb_proc,host=self.host,log=logsrun)
            run_subprocess(casepath,ALYA_SETS,'%s-boundary.nsi.set 3'%self.case,log=logssets)
            
        elif which == 'execute':
            casepath = os.path.join('alya_files','%s'%self.host,'EP_%d'%self.episode_number)
            logsrun  = os.path.join('logs','log_last_execute_run.log')
            logssets = os.path.join('logs','log_sets.log')
            # Run subprocess
            run_subprocess(casepath,'mkdir -p','logs') # Create logs folder
            run_subprocess(casepath,ALYA_BIN,'%s'%self.case,nprocs=nb_proc,host=self.host,log=logsrun)
            run_subprocess(casepath,ALYA_SETS,'%s-boundary.nsi.set 3'%self.case,log=logssets)
              
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
              
    def switch(self, state):
        if self.do_baseline == False:
            return
            
        # Open file
        filepath = os.path.join('alya_files','baseline') if self.episode_number == 0 else os.path.join('alya_files','%s'%self.host,'EP_%d'%self.episode_number)
        runtype  = 'CONTI' if state == 'continue' else 'NOCONTI'
        write_run_type(filepath,runtype,freq=1000)
            
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
   
    def save_history_parameters(self, nb_actuations):
        
        cr_start('ENV.save_cd_cl',0)
        
        # Save at the end of every episodes
        self.episode_drags = np.append(self.episode_drags, self.history_parameters["drag"])
        self.episode_lifts = np.append(self.episode_lifts, self.history_parameters["lift"])
        
        if self.action_count == nb_actuations or self.episode_number == 0:
            print("Action : saving history parameters in saved_models/output.csv")
            self.last_episode_number = self.episode_number
            
            avg_drag = np.mean(self.history_parameters["drag"][-1:])
            avg_lift = np.mean(self.history_parameters["lift"][-1:])
            
            if self.do_baseline == True:
                name = "output.csv"
                if(not os.path.exists("saved_models")):
                    os.mkdir("saved_models")
                if(not os.path.exists("saved_models/"+name)):
                    with open("saved_models/"+name, "w") as csv_file:
                        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                        spam_writer.writerow(["Episode", "AvgDrag", "AvgLift"])
                        spam_writer.writerow([self.last_episode_number, avg_drag, avg_lift])
                else:
                    with open("saved_models/"+name, "a") as csv_file:
                        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                        spam_writer.writerow([self.last_episode_number, avg_drag, avg_lift])
            self.episode_drags = np.array([])
            self.episode_lifts = np.array([])
            
            if self.do_baseline == True:
                # Writes all the cl and cd in .csv
                # IS THIS NECESSARY? I THINK WE DO NOT USE THE BEST MODEL
                if(os.path.exists("saved_models/output.csv")):
                    if(not os.path.exists("best_model")):
                        shutil.copytree("saved_models", "best_model")
                    else:
                        with open("saved_models/output.csv", 'r') as csvfile:
                            data = csv.reader(csvfile, delimiter = ';')
                            for row in data:
                                lastrow = row
                            last_iter = lastrow[1]
                        with open("best_model/output.csv", 'r') as csvfile:
                            data = csv.reader(csvfile, delimiter = ';')
                            for row in data:
                                lastrow = row
                            best_iter = lastrow[1]
                        if float(best_iter) < float(last_iter):
                            print("best_model updated")
                            if(os.path.exists("best_model")):
                                shutil.rmtree("best_model")
                            shutil.copytree("saved_models", "best_model")
                        
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
        
        if(not os.path.exists("actions/{}/ep_{}/".format(self.host, self.episode_number))):
            os.mkdir("actions/{}/ep_{}/".format(self.host, self.episode_number))
        
        path_a = "actions/{}/ep_{}/".format(self.host, self.episode_number)
        
        action_line = "{}".format(self.action_count)
        for i in range(self.n_jets):
            action_line = action_line + "; {}".format(self.action[i])
        
        if(not os.path.exists(path_a+name_a)):
            header_line = "Action"
            for i in range(self.n_jets):
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
            
        if(not os.path.exists("rewards/{}/ep_{}/".format(self.host, self.episode_number))):
            os.mkdir("rewards/{}/ep_{}/".format(self.host, self.episode_number))
            
        path_a = "rewards/{}/ep_{}/".format(self.host, self.episode_number)
        
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
            
        path_a = "final_rewards/{}/".format(self.host)
        
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
       
    ### AQUI DEBEMOS ANULAR EL RESUPERAR EL BASELINE SI YA EXISTE EL QUE TOCA
    def recover_start(self):
        runpath = 'alya_files'
        runbin  = 'cp -r'
        runargs = 'baseline %s'%os.path.join('%s'%self.host,'EP_%d'%self.episode_number)
        logs    = os.path.join('baseline','logs','log_restore_last_episode.log')
        run_subprocess(runpath,runbin,runargs,log=logs)
        
        self.action = np.zeros(self.n_jets)
         
        #lowcost mode
        if(self.episode_number>1):
            runbin  = 'rm -r'
            runargs = os.path.join('%s'%self.host,'EP_%d'%(self.episode_number-1))
            #avoid checks in deterministic
            if self.deterministic == False:
               run_subprocess(runpath,runbin,runargs)
    
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    # create folder for each cpu id in parallel
    def create_cpuID(self):
        runpath = 'alya_files'
        runbin  = 'mkdir'
        if self.deterministic == False:
            runargs = self.host
            run_subprocess(runpath,runbin,runargs)
            
            # Write the nodes running this environmment
            name = "nodes"
            if(not os.path.exists("alya_files/{}/".format(self.host)+name)):
                with open("alya_files/{}/".format(self.host)+name, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Nodes in this learning"])
                    spam_writer.writerow([self.node])
            else:
                with open("alya_files/{}/".format(self.host)+name, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Nodes in this learning"])
                    spam_writer.writerow([self.node])
        else:
            runargs = 'deterministic'
            run_subprocess(runpath,runbin,runargs,check_return=False)
            
        print('Folder created for CPU ID: ', self.host)
   
             
    # Optional
    def close(self):
        super().close()

    
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    ## Default function required for the DRL
    
    def states(self):

        return dict(type='float',
                    shape=(len(self.output_params["locations"]), )
                    )
            
   #-----------------------------------------------------------------------------------------------------
   #-----------------------------------------------------------------------------------------------------    
        
    def actions(self):
        
        """Action is a list of n_jets-1 capped values of Q"""
        
        return dict(type='float',
                    shape=(self.n_jets-1), 
                           min_value=self.optimization_params["min_value_jet_MFR"],
                           max_value=self.optimization_params["max_value_jet_MFR"]
                    )
    
    #-----------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------
    
    
    def reset(self):
        """Reset state"""
        
        # Create a folder for each environment
        if self.check_id == True:
            self.create_cpuID()
            self.check_id = False
        
        # Clean
        print("\n\nLocation: Reset")
        print("Action: start to set up the case, set the initial conditions and clean the action counter")
        self.clean(False)
        
        # Apply new time frame
        self.simulation_timeframe = simulation_params["simulation_timeframe"]
        t1 = self.simulation_timeframe[0]
        t2 = self.simulation_timeframe[1]
        self.simulation_timeframe = [t1,t2]
        print("The actual timeframe is between {} and {}: ".format(t1,t2))
        
        # Advance in episode
        self.episode_number += 1
        if self.deterministic == True:
            # If we are in deterministic, the directory will be 'deterministic'
            self.host = 'deterministic'
        
        # Copy the baseline in the environment directory     
        if self.action_count == 1:
            self.recover_start()
        
        write_time_interval(os.path.join('alya_files','%s'%self.host,'EP_%d'%self.episode_number),t1,t2)

        print("Actual episode: {}".format(self.episode_number))
        print("\n\Action: extract the probes")
        NWIT_TO_READ=1 # Read n timesteps from witness file from behind, last instant
        filename     = os.path.join('alya_files','%s'%self.host,'EP_%d'%self.episode_number,'%s.nsi.wit'%self.case)
        probes_value = read_last_wit(filename,output_params["probe_type"], self.optimization_params["norm_press"],NWIT_TO_READ)
        
        return probes_value
        
    #-----------------------------------------------------------------------------------------------------
   
    def execute(self, actions):
        
        action = []
        for i in range(self.n_jets-1):
            action.append(self.optimization_params["norm_Q"]*actions[i])
        action.append(-sum(action))  # This is to ensure 0 mass flow rate in the jets
        
        self.previous_action = self.action #save the older one to smooth the change
        self.action = action #update the new to reach at the end of smooth
        print('New flux computed :\n\tQs : ', self.action)
        
        self.last_time = self.simulation_timeframe[1]
        t1 = round(self.last_time,3)
        t2 = t1 + self.delta_t_smooth
            
        self.simulation_timeframe = [t1,t2]
        write_time_interval(os.path.join('alya_files','%s'%self.host,'EP_%d'%self.episode_number),t1,t2)
        
        simu_path = os.path.join('alya_files','%s'%self.host,'EP_%d'%self.episode_number)
        if case == 'cylinder':
            for ijet, jet in enumerate(self.Jets.values()): # Only need to loop on the values, i.e., Jet class
                # Update jets for the given epoch
                jet.update(
                    self.previous_action[ijet],  # Q_pre
                    self.action[ijet],           # Q_new
                    self.simulation_timeframe[0] # time_start
                )
                # Update the jet profile alya file
                jet.update_file(simu_path)
        elif case == 'airfoil':
            for ijet, jet in enumerate(self.Jets.values()): # Only need to loop on the values, i.e., Jet class
                # Update jets for the given epoch
                if self.smooth_func == 'parabolic':
                    self.slope_pre = jet.slope_new
                else:
                    self.slope_pre = 0
                    
                jet.update_airfoil(
                    self.previous_action[ijet],  # Q_pre
                    self.action[ijet],           # Q_new
                    self.simulation_timeframe[0], # time_start
                    self.slope_pre
                )
                # Update the jet profile alya file
                jet.update_file(simu_path)
        
        # Start an alya run
        t0 = time.time()
        print("\n\nLocation : Execute/SmoothControl\nAction: start a run of Alya")
        self.run(which = 'execute')
        print("Done. time elapsed : ", time.time() - t0)
        
        # Get the new avg drag and lift
        average_drag, average_lift = compute_avg_lift_drag(self.episode_number, cpuid = self.host)
        self.history_parameters["drag"].extend([average_drag])
        self.history_parameters["lift"].extend([average_lift])
        self.history_parameters["time"].extend([self.last_time])
        self.history_parameters["episode_number"].extend([self.episode_number])
        self.save_history_parameters(nb_actuations)
        
        # Write the action
        self.save_this_action()
        
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
        
        # Read witness file from behind, last instant
        NWIT_TO_READ=1
        filename      = os.path.join('alya_files','%s'%self.host,'EP_%d'%self.episode_number,'%s.nsi.wit'%self.case)
        probes_values = read_last_wit(filename,output_params["probe_type"], self.optimization_params["norm_press"],NWIT_TO_READ)

        return probes_values, terminal, reward
    
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
            
            ## le añadimos el offset de 3.21 para partir de reward nula y que solo vaya a (+)
            return self.optimization_params["norm_reward"] * (- avg_drag_2 - self.optimization_params["penal_cl"] * abs(avg_lift_2) + self.optimization_params["offset_reward"])
        
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
