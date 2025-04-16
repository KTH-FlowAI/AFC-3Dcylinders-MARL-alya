import numpy as np
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()

num_re = '3900'
size_title = 14
linewidth_legend = 2

cd_base = 1.25
cl_base = 0.0
cd_deter = 1.18
cl_deter = -0.11
#cd_2D   = 1.21

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2)

data = np.genfromtxt("../saved_models/output.csv", delimiter=";", names=["Episode", "AvgDrag", "AvgLift","AvgDrag_GLOBAL","AvgLift_GLOBAL"])

axs[0,0].plot(data['AvgDrag'], color ='k', linewidth=0.5, alpha = 0.5)
axs[0,0].plot(range(12,len(data['AvgDrag'])-2), np.convolve(data['AvgDrag'],np.ones(15), 'valid' )/15, 'g', linewidth=2)
#axs[0,0].set_title("Local Cd during (Re = %s)" %num_re, fontsize=size_title)
axs[0,0].set_xlabel("Pseudoenvironments episodes", fontsize=size_title)
axs[0,0].set_ylabel("Cd local", fontsize=size_title)
axs[0,0].tick_params(axis='both',labelsize=size_title)
axs[0,0].axhline(y=cd_base, color='k', linestyle='--', label='Baseline           Cd = %0.2f' %cd_base, linewidth=linewidth_legend)
#axs[0,0].axhline(y=cd_2D, color='m', linestyle='--',   label='2D training        Cd = %0.2f' %cd_2D, linewidth=linewidth_legend)
axs[0,0].axhline(y=cd_deter, color='r', linestyle='--',label='3D Deterministic Cd = %0.2f' %cd_deter, linewidth=linewidth_legend)
axs[0,0].legend(loc='best', fancybox=True)
axs[0,0].grid()
#axs[0,0].gcf().suptitle('fig1')
#axs[0,0].tight_layout()
#axs[0,0].show()
plt.savefig("cd_plot_re%s_3D_MARL.png" %num_re)

axs[0,1].plot(data['AvgLift'], color ='k', linewidth=0.5, alpha = 0.5)
axs[0,1].plot(range(12,len(data['AvgDrag'])-2), np.convolve(data['AvgLift'],np.ones(15), 'valid' )/15, 'r', linewidth=2)
#axs[0,1].set_title("Local Cl training (Re = %s)" %num_re, fontsize=size_title)
axs[0,1].set_xlabel("Pseudoenvironments episodes", fontsize=size_title)
axs[0,1].set_ylabel("Cl local", fontsize=size_title)
axs[0,1].tick_params(axis='both',labelsize=size_title)
axs[0,1].axhline(y=0, color='k', linestyle='--',        label='Baseline           Cl = %0.2f' %cl_base, linewidth=linewidth_legend)
#axs[0,1].axhline(y=0, color='m', linestyle='--',        label='2D training        Cl = %0.2f' %cl_2D, linewidth=linewidth_legend)
axs[0,1].axhline(y=cl_deter, color='r', linestyle='--', label='3D Deterministic Cl = %0.2f' %cl_deter, linewidth=linewidth_legend)
axs[0,1].legend(loc='best', fancybox=True)
axs[0,1].grid()
#axs[0,1].gcf().suptitle('fig2')
#plt.tight_layout()
#plt.show()
plt.savefig("cl_plot_re%s_3D_MARL.png" %num_re)

#plt.plot(data['AvgLift_GLOBAL'], 'k--', linewidth=0.5)
axs[1,1].plot(np.arange(7*0.1,len(data['AvgLift_GLOBAL'])*0.1-0.2,0.1), np.convolve(data['AvgLift_GLOBAL'],np.ones(10), 'valid' )/10, 'c', linewidth=2)
#axs[1,1].set_title("Global Cl training (Re = %s)" %num_re, fontsize=size_title)
axs[1,1].set_xlabel("CFD runs", fontsize=size_title)
axs[1,1].set_ylabel("Cl global", fontsize=size_title)
axs[1,1].tick_params(axis='both',labelsize=size_title)
axs[1,1].axhline(y=0, color='k', linestyle='--', label='Baseline           Cl = %0.2f' %cl_base, linewidth=linewidth_legend)
#axs[1,1].axhline(y=0, color='m', linestyle='--', label='2D training        Cl = %0.2f' %cl_2D, linewidth=linewidth_legend)
axs[1,1].axhline(y=cl_deter, color='r', linestyle='--', label='3D Deterministic Cl = %0.2f' %cl_deter, linewidth=linewidth_legend)
axs[1,1].legend(loc='best', fancybox=True)
axs[1,1].grid()
#axs[1,1].gcf().suptitle('fig3')
#plt.tight_layout()
#plt.show()
plt.savefig("cl_plot_GLOBAL_re%s_3D_MARL.png" %num_re)

#plt.plot(data['AvgDrag_GLOBAL'], 'k--', linewidth=0.5)
axs[1,0].plot(np.arange(7*0.1,len(data['AvgDrag_GLOBAL'])*0.1-0.2,0.1), np.convolve(data['AvgDrag_GLOBAL'],np.ones(10), 'valid' )/10, 'm', linewidth=2)
#axs[1,0].set_title("Global Cd training (Re = %s)" %num_re, fontsize=size_title)
axs[1,0].set_xlabel("CFD runs", fontsize=size_title)
axs[1,0].set_ylabel("Cd global", fontsize=size_title)
axs[1,0].tick_params(axis='both',labelsize=size_title)
axs[1,0].axhline(y=cd_base, color='k', linestyle='--', label='Baseline           Cd = %0.2f' %cd_base, linewidth=linewidth_legend)
#axs[1,0].axhline(y=cd_2D, color='m', linestyle='--',   label='2D training        Cd = %0.2f' %cd_2D, linewidth=linewidth_legend)
axs[1,0].axhline(y=cd_deter, color='r', linestyle='--',label='3D Deterministic Cd = %0.2f' %cd_deter, linewidth=linewidth_legend)
axs[1,0].legend(loc='best', fancybox=True)
axs[1,0].grid()
#axs[1,0].gcf().suptitle('fig4')
#plt.tight_layout()
#plt.show()
plt.savefig("cd_plot_GLOBAL_re%s_3D_MARL.png" %num_re)

plt.tight_layout()

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.2, hspace=0.2)

# Set the figure size to accommodate the subplots without overlapping
fig.set_size_inches(14, 8)

plt.show()
