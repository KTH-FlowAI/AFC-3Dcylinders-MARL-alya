import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as mtick
import numpy as np
import math
import os

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', labelsize=14)
plt.rc('font', size=9)
plt.rc('legend', fontsize=9)               # Make the legend/label fonts 
plt.rc('xtick', labelsize=12)               # a little smaller
plt.rc('ytick', labelsize=12)
plt.rcParams['mathtext.fontset']='cm'

##############################################

### PLOT TOTAL REWARDS AND CL, CD CONTRIBUTION

# need to enter the batch and which Re number

##############################################

nbatches = 7

def moving_rms(data, window_size):
    diff = data[windowsize-3:len(data)-2] - np.convolve(data,np.ones(window_size), 'valid' )/window_size
    print(type(diff),diff.size)

    return np.sqrt(np.convolve(diff**2, np.ones(window_size), mode='valid')/(window_size-1))



num_re = 3900

size_title = 20
linewidth_legend = 1.5

linewidth_coef = 2

cd_base = 1.05
cl_base = 0
cd_deter = 1.05
cl_deter = 0.0
    
data = np.genfromtxt("../saved_models/output.csv", delimiter=";", names=["Episode", "AvgDrag", "AvgLift","AvgDrag_GLOBAL","AvgLift_GLOBAL"])

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, sharex=True)
fig.set_size_inches(8, 5)

axs[0].plot(data['AvgDrag'], color ='k', linewidth=0.5, alpha = 0.25)
axs[0].plot(range(12,len(data['AvgDrag'])-2), np.convolve(data['AvgDrag'],np.ones(15), 'valid' )/15, 'g', linewidth=3)
axs[0].set_ylabel("Cd")
axs[0].set_xlim(0,axs[0].set_xlim()[1])
#axs[0].set_ylim(1.0,1.3)
axs[0].set_xticklabels([])
axs[0].axhline(y=cd_base, color='k', linestyle='--', label='Baseline             Cd = %0.2f' %cd_base, linewidth=linewidth_legend)
axs[0].axhline(y=cd_deter, color='r', linestyle='--',label='3D Deterministic Cd = %0.2f' %cd_deter, linewidth=linewidth_legend)
axs[0].legend(fancybox = True, facecolor='white', edgecolor='none',ncol = 2)
axs[0].grid()


axs[1].plot(data['AvgLift'], color ='k', linewidth=0.5, alpha = 0.25)
axs[1].plot(range(12,len(data['AvgDrag'])-2), np.convolve(data['AvgLift'],np.ones(15), 'valid' )/15, 'r', linewidth=3)
axs[1].set_ylabel("Cl")
axs[1].set_xlim(0,axs[0].set_xlim()[1])
axs[1].set_xticklabels([])
axs[1].axhline(y=0, color='k', linestyle='--',        label='Baseline             Cl = %0.2f' %cl_base, linewidth=linewidth_legend)
axs[1].axhline(y=cl_deter, color='r', linestyle='--', label='3D Deterministic Cl = %0.2f' %cl_deter, linewidth=linewidth_legend)
axs[1].legend(fancybox = True, facecolor='white', edgecolor='none',ncol = 2)
axs[1].grid()

fig.tight_layout()

# Adjust the spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.2)

# Adjust the spacing between subplots
fig.tight_layout()
fig.subplots_adjust(wspace=0.2, hspace=0.1,bottom=0.15)
fig.savefig("FTaC_cdcl_case8_%d.png" %num_re)
fig.show()


# Create a 2x2 grid of subplots
fig_rew, ax_rew = plt.subplots(1, figsize=(5, 5), sharex=True)
    
y = (0.8*(5*(np.abs(data['AvgDrag']-cd_base)-np.abs(0.6*data['AvgLift'])))+0.2*(5*(np.abs(data['AvgDrag_GLOBAL']-cd_base)-0.6*np.abs(data['AvgLift_GLOBAL']))))
print(type(y),y.size)

ax_rew.plot(range(12,len(data['AvgDrag'])-2), np.convolve(y,np.ones(15), 'valid' )/15, 'k', linewidth=linewidth_coef, label='Total reward')
#ax_rew.plot(range(len(data['AvgDrag'])), y, '--', linewidth=linewidth_coef, label='Total reward RAW')

windowsize=15
std_moving = moving_rms(y, windowsize)
data_std = np.convolve(y,np.ones(windowsize), 'valid' )/windowsize
print(type(data_std),data_std.size)
print(type(std_moving),std_moving.size)
ax_rew.fill_between(range(15+windowsize,len(std_moving)+15+windowsize), data_std[windowsize-1:len(data)] + std_moving, data_std[windowsize-1:len(data)] - std_moving, alpha=0.2, color='k', label = 'Standard deviation')


#axs[3].plot(0.8*(5*(np.abs(data['AvgDrag']-cd_base)-0.6*data['AvgLift']))+0.2*(5*(np.abs(data['AvgDrag_GLOBAL']-cd_base)-0.6*data['AvgLift_GLOBAL'])), color ='k', linewidth=0.5, alpha = 0.5, label = 'Cd contribution')
ax_rew.plot(range(12,len(data['AvgDrag'])-2), np.convolve(0.8*(5*(np.abs(data['AvgDrag']-cd_base)))+0.2*(5*(np.abs(data['AvgDrag_GLOBAL']-cd_base))),np.ones(15), 'valid' )/15, 'red', linewidth=linewidth_coef, label = '$C_d$ contribution', alpha = 0.5)

ax_rew.plot(range(12,len(data['AvgDrag'])-2), np.convolve(0.8*(5*(-0.6*np.abs(data['AvgLift'])))+0.2*(5*(-np.abs(0.6*data['AvgLift_GLOBAL']))),np.ones(15), 'valid' )/15, 'blue', linewidth=linewidth_coef, label = '$C_l$ contribution', alpha = 0.5)

#ax_rew.set_xlim(0,200)
#axs[2].set_ylim(-1.0,1.3)
ax_rew.tick_params(axis='both')
ax_rew.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax_rew.set_ylabel(r"$R$")
ax_rew.axhline(y=0, color='k', linestyle='--', linewidth=linewidth_legend)

ax_rew.grid()

ax_rew.set_xlabel("MARL episodes")
ax_rew.legend(fancybox = True, facecolor='white', edgecolor='none',ncol = 2,loc='upper center', bbox_to_anchor=(0.5, -0.2))


# Adjust the spacing between subplots
fig_rew.tight_layout()
fig_rew.subplots_adjust(wspace=0.2, hspace=0.1,bottom=0.3)
fig_rew.savefig("FTaC_rewards_case8_%d.png" %num_re)
fig_rew.show()

plt.show()

   

    
