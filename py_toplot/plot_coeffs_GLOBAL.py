import numpy as np
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()


data = np.genfromtxt("../saved_models/output.csv", delimiter=";", names=["Episode", "AvgDrag", "AvgLift","AvgDrag_GLOBAL","AvgLift_GLOBAL"])

plt.plot(data['AvgDrag'], 'b--', linewidth=0.5, label='Instantaneous learning')
plt.plot(range(12,len(data['AvgDrag'])-2), np.convolve(data['AvgDrag'],np.ones(15), 'valid' )/15, 'g', linewidth=2, label='Averaged learning over the last 15 episodes')
plt.title("CD vs episodes")
plt.xlabel("#episodes")
plt.ylabel("Cd")
#plt.axhline(y=3.17, color='r', linestyle='--', label='Baseline')
plt.legend(loc='best', fancybox=True)
plt.grid()
plt.show()
plt.savefig("cd_plot.png")

plt.plot(data['AvgLift'], 'b--', linewidth=0.5, label='Instantaneous learning')
plt.plot(range(12,len(data['AvgDrag'])-2), np.convolve(data['AvgLift'],np.ones(15), 'valid' )/15, 'r', linewidth=2, label='Averaged learning over the last 15 episodes')
plt.title("Cl vs episodes")
plt.xlabel("#episodes")
plt.ylabel("Cl")
#plt.axhline(y=0, color='r', linestyle='--', label='0 lift')
plt.legend(loc='best', fancybox=True)
plt.grid()
plt.show()
plt.savefig("cl_plot.png")

plt.plot(data['AvgLift_GLOBAL'], 'b--', linewidth=0.5, label='Instantaneous learning')
plt.plot(range(7,len(data['AvgDrag_GLOBAL'])-2), np.convolve(data['AvgLift_GLOBAL'],np.ones(10), 'valid' )/10, 'c', linewidth=2, label='Averaged learning over the 10 invariants')
plt.title("Cl vs episodes")
plt.xlabel("#episodes")
plt.ylabel("Cl")
#plt.axhline(y=0, color='r', linestyle='--', label='0 lift')
plt.legend(loc='best', fancybox=True)
plt.grid()
plt.show()
plt.savefig("cl_plot_GLOBAL.png")

plt.plot(data['AvgDrag_GLOBAL'], 'b--', linewidth=0.5, label='Instantaneous learning')
plt.plot(range(7,len(data['AvgDrag_GLOBAL'])-2), np.convolve(data['AvgDrag_GLOBAL'],np.ones(10), 'valid' )/10, 'm', linewidth=2, label='Averaged learning over the 10 invariants')
plt.title("Cd vs episodes")
plt.xlabel("#episodes")
plt.ylabel("Cd")
#plt.axhline(y=0, color='r', linestyle='--', label='0 lift')
plt.legend(loc='best', fancybox=True)
plt.grid()
plt.show()
plt.savefig("cd_plot_GLOBAL.png")

