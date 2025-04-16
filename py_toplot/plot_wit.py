import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math

plt.close("all")

plt.style.use('classic')
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

NZONES=151
MYZONE=5
tt=np.array([])

filepath = '../alya_files/environment_1/EP_1/cylinder.nsi.wit'

U=np.array([])
V=np.array([])
W=np.array([])
P=np.array([])

with open(filepath) as fp:  
   for cnt, line in enumerate(fp):
       if line[0] == '#' :
           if line[0:6] == '# Time':
               stripLine = line.strip().split('=')
               time = float(stripLine[1])
               tt = np.append(tt,time)
               q = [0,0,0,0]                  
               cntCheck = cnt + NZONES
               continue
           else:
               continue
       if cntCheck + 1 > cnt:
          stripLine = line.strip().split(' ')
          stripLine = [x for x in stripLine if x != '']
          if(int(stripLine[0])==MYZONE) : # wind
              q[0] = q[0] + float(stripLine[2-1]) # U
              q[1] = q[1] + float(stripLine[3-1]) # V
              q[2] = q[2] + float(stripLine[4-1]) # W
              q[3] = q[3] + float(stripLine[5-1]) # P
          if(int(stripLine[0])==NZONES) :
              U = np.append(U,q[0])
              V = np.append(V,q[1])
              W = np.append(W,q[2])
              P = np.append(P,q[3])
              continue


                  


# U
fig=plt.figure(1, figsize=(7.5, 5), dpi=80)
plt.plot(tt,U,'k',linewidth=4.0)
plt.ylabel(r'U')
plt.xlabel(r't')
#plt.axis([0.0, 25, 1.3, 1.37])
plt.tight_layout()
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(3)
plt.show()


# V
fig=plt.figure(2, figsize=(7.5, 5), dpi=80)
plt.plot(tt,V,'k',linewidth=2.0)
plt.ylabel(r'V')
plt.xlabel(r't')
#plt.axis([0.0, 25, 1.3, 1.37])
plt.tight_layout()
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
plt.show()


# W
fig=plt.figure(3, figsize=(7.5, 5), dpi=80)
plt.plot(tt,W,'k',linewidth=2.0)
plt.ylabel(r'W')
plt.xlabel(r't')
#plt.axis([0.0, 25, 1.3, 1.37])
plt.tight_layout()
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
plt.show()


# P
fig=plt.figure(4, figsize=(7.5, 5), dpi=80)
plt.plot(tt,P,'k',linewidth=3.0)
plt.ylabel(r'P')
plt.xlabel(r't')
#plt.axis([0.0, 25, 1.3, 1.37])
plt.tight_layout()
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(2)
plt.show()
