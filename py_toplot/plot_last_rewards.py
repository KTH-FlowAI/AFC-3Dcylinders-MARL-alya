import numpy as np
import matplotlib.pyplot as plt
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../")
from parameters import num_servers

for i in range(num_servers):  # Uncomment this line to plot all environments
#for i in range(1,2):  # Change the second component in range() to set how many episodes do you want to see
    data0 = np.genfromtxt("../final_rewards/environment{}/{}_1/output_final_rewards.csv".format(i+1), delimiter=";", names=["EPISODE", "REWARD"])
    data = {}
    data["EPISODE"] = []
    data['REWARD'] = []
    for j in range(len(data0)):
        data["EPISODE"].append(data0[j][0])
        data["REWARD"].append(data0[j][1])
        
    plt.plot(data['EPISODE'], data['REWARD'])

plt.plot(data['EPISODE'], data['REWARD'])
plt.title("Final rewards at episodes")
plt.xlabel("#episodes")
plt.ylabel("Final reward")
plt.grid()
plt.show()
plt.savefig("plot_final_rewards.png")
