import matplotlib.pyplot as pyplot
import numpy as np
import time
#input from files...
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

raw_data = np.genfromtxt(fname="train.txt",delimiter=',',unpack=True)
raw2 = np.genfromtxt(fname="test.txt",delimiter=",",unpack=True)

labels = []
lb = []
data = []
d2 = []

for re in raw2:
    lb.append(int(re[0]))
    d2.append(re[1:])

for row in raw_data:
    labels.append(int(row[0]))
    data.append(row[1:])

lol = np.arange(0.0,13.0,1.0)
lol2 = np.arange(0.0,9.0,1.0)
fig, ax = pyplot.subplots() # pyplot.figure()


arr = np.zeros(shape=(len(data),len(data)))
arr2 = []
arr3 = []
x = []
x2 = []

#print str(len(data))
for r in range(len(data)):
    for r2 in range(len(data[r])):
        x.append(r)
        arr2.append(data[r][r2])

    #print "r : " + str(r) + "data: " + str(data[r])
#        da = data[r][r2]
#        print da
    #2d array, maa finen x verdiene og legge disse og telle disse feks:
    #[1][10] <- 1 er X verdien, og 10 er valuen. 10 kommer fra antall forekomster av
    #for r2 in range(len(da)):

for r3 in range(len(d2)):
    for r4 in range(len(d2[r3])):
        arr3.append(d2[r3][r4])



for d in range(len(x)-len(arr3)):
    arr3.append(0)

#print len(x)
#for r in range(len(d2)):
 #   da2 = d2[r]
  #  arr2.append(da2)
if(arr2 < arr3):
    print "arr2 er lengst"
#pyplot.stackplot(lol,arr)
#print arr2
pyplot.glyphplot(x,arr3,arr2)

#print da1
#print da2
#pyplot.stackplot(lol,da1,da2)

pyplot.show(block=True)
