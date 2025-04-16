import numpy as np
import matplotlib.pyplot as plt
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../")
from parameters import simulation_params, num_servers, n_jets

# Get the last episode
episode_list = next(os.walk('../alya_files/environment_1/'))[1]
num_episode = episode_list[0][-1]
#num_episode = 1  # Uncomment this line to select a desired episode

simulation_duration = simulation_params["simulation_duration"]
T_smoo              = simulation_params["delta_t_smooth"]
smooth_func         = simulation_params["smooth_func"]
names = ["Action"]
for i in range(n_jets):
    names.append("Jet_{}".format(i+1))

plot_colors = ['b','g','r','k']
#for i in range(0,num_servers):  # Uncomment this line to plot all environments
for i in range(1,2):  # Change the second component in range() to set how many episodes do you want to see
    string_epi = 'ep_{}'.format(num_episode)
    data0 = np.genfromtxt("../actions/environment_{}/{}/output_actions.csv".format(i, string_epi), delimiter=";", names=names)
    for j in range(n_jets+1):
        data0[0][j] = 0
    
    data = {}
    data["Action"] = [0]
    for j in range(n_jets):
        data['Jet_{}'.format(j+1)] = [0]
    if smooth_func == 'linear':
        for j in range(len(data0)):
            data["Action"].append(simulation_duration + j*T_smoo)
            for k in range(n_jets):
                data["Jet_{}".format(k+1)].append(data0[j][k+1])
            
    elif smooth_func == 'parabolic':
        slope_pre = np.zeros(n_jets)
        data_lin = {}
        data_lin["Action"] = [0]
        for j in range(n_jets):
            data_lin['Jet_{}'.format(j+1)] = [0]
        for j in range(len(data0)-1):
            time_start = simulation_duration + j*T_smoo
            t = np.linspace(time_start, time_start+T_smoo, 50)
            for l in range(len(t)):
                data["Action"].append(t[l])
            for k in range(n_jets):
                delta_Q = data0[j+1][k+1]-data0[j][k+1]
                a = delta_Q/T_smoo**2 - slope_pre[k]/T_smoo
                b = (2*time_start+T_smoo)/T_smoo*slope_pre[k] - 2*time_start/T_smoo**2*delta_Q
                c = data0[j+1][k+1] + (time_start-T_smoo)*(time_start+T_smoo)/T_smoo**2*delta_Q \
                                  - (time_start+T_smoo)*time_start/T_smoo*slope_pre[k]
            
                action = a*t**2 + b*t + c
                
                for l in range(len(t)):
                    data["Jet_{}".format(k+1)].append(action[l])
            
                slope_pre[k] = 2*a*(time_start+T_smoo) + b
            
            # Compare with linear
            data_lin["Action"].append(simulation_duration + j*T_smoo)
            for k in range(n_jets):
                data_lin["Jet_{}".format(k+1)].append(data0[j][k+1])
                plt.plot(data_lin['Action'], data_lin['Jet_{}'.format(k+1)], color='{}'.format(plot_colors[k]),linestyle='dashed')
    
    for j in range(n_jets):
        plt.plot(data['Action'], data['Jet_{}'.format(j+1)], color='{}'.format(plot_colors[j]), label='Jet_{}'.format(j+1))

plt.title("Q variation during episode {}".format(num_episode))
plt.xlim([data["Action"][1] - T_smoo, data["Action"][-1]])
plt.xlabel("#action")
plt.ylabel("Q jet value")
plt.grid()
plt.legend(loc='best', fancybox=True)
plt.show()
plt.savefig("actions_plot.png")
