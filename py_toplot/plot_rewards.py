import numpy as np
import matplotlib.pyplot as plt
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../")
from parameters import simulation_params, num_servers

# Get the last episode
episode_list = next(os.walk('../alya_files/environment1/'))[1]
num_episode = episode_list[0][-1]
#num_episode = 1  # Uncomment this line to select a desired episode

simulation_duration = simulation_params["simulation_duration"]
T_smoo              = simulation_params["delta_t_smooth"]
for i in range(1,num_servers+1):  # Uncomment this line to plot all environments
#for i in range(1,2):  # Change the second component in range() to set how many episodes do you want to see
    string_epi = 'ep_{}'.format(num_episode)
    data0 = np.genfromtxt("../rewards/environment{}/{}/output_rewards.csv".format(i, string_epi), delimiter=";", names=["Action", "Reward"])
    data = {}
    data["Action"] = []
    data['Reward'] = []
    for j in range(len(data0)):
        data["Action"].append(simulation_duration + j*T_smoo)
        data["Reward"].append(data0[j][1])
        
    plt.plot(data['Action'], data['Reward'])

plt.title("REWARDS during episode {}".format(num_episode))
plt.xlabel("#action")
plt.ylabel("Reward")
plt.grid()
plt.show()
plt.savefig("reward_plot.png")
