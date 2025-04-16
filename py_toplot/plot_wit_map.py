import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
#from mpl_toolkits.mplot3d.art3D import Cylinder
import sys 
import csv

PATH_ALYA = sys.argv[1]

# load witness points 
wit_coord = pd.read_csv(PATH_ALYA + 'witness.csv')
x = wit_coord["x"]
y = wit_coord["y"]
z = wit_coord["z"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)

for xx,yy,zz,i in zip(x,y,z,range(len(x))):
	ax.text(xx+.1, yy+.1, zz, i)

ax.set_xlabel("x coord")
ax.set_ylabel("y coord")
ax.set_zlabel("z coord")

ax.set_xlim(0,21)
ax.set_ylim(0,15)
ax.set_zlim(0,4.1)

# add cylinder
#c = Cylinder((7.5, 7.5, 0), 15, 0, alpha=0.5)
#ax.add_artist(c)

circle1 = plt.Circle((7.5,7.5,2), 0.5, color='r')
circle2 = plt.Circle((7.5,7.5,2.1), 0.5, color='r')
circle3 = plt.Circle((7.5,7.5,1.9), 0.5, color='r')

plt.show()

print("POOOL --> wit_coord: %s" %wit_coord)

