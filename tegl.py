import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from array import array

# Data to be represented
# ----------
labels = ['1', '2', '3', '4', '5', '6',
          '7', '8', '9', '10', '11', '12', '13']
#n = len(labels)
data = np.random.uniform(0,0,13)

data2 = np.genfromtxt(fname="train.txt",delimiter=',',unpack=True)
#labels = []
for r in range(len(data2)):
    for r2 in range(len(data2[r])):
        #labels.append(r)
        data[r-1] += data2[r][r2]


print data
n = len(labels)


# ----------

# Make figure square and background the same colors as axes (white)
fig = plt.figure(figsize=(8,6), facecolor='white')

# Make a new polar axis
axes = plt.subplot(111, polar=True, axisbelow=True)

# Put labels on outer
T = np.arange(np.pi/n, 2*np.pi, 2*np.pi/n)
R = np.ones(n)*10
width = 2*np.pi/n

# Label background
bars  = axes.bar(T, R, width=width, bottom=9,
                 linewidth = 2, facecolor = '0.9', edgecolor='1.00')
# Labels
for i in range(T.size):
    theta = T[n-1-i]+np.pi/n + np.pi/2
    plt.text(theta, 9.5, labels[i], rotation=180*theta/np.pi-90,
             family='Helvetica Neue', size=7,
             horizontalalignment="center", verticalalignment="center")

# Data
R = 1 + data*6
bars = axes.bar(T,R, width=width, bottom=2, linewidth=1, facecolor = '0.75', edgecolor='1.00')
for i,bar in enumerate(bars):
    bar.set_facecolor(plt.cm.hot(R[i]/10))


# Set ticks, tick labels and grid
plt.ylim(0,10)
plt.xticks(T)
plt.yticks(np.arange(2,9))
axes.grid(which='major', axis='y', linestyle='-', color='0.75')
axes.grid(which='major', axis='x', linestyle='-', color='1.00')
for theta in T:
    axes.plot([theta,theta], [4,9], color='w', zorder=2, lw=1)
axes.set_xticklabels([])
axes.set_yticklabels([])

plt.show()
