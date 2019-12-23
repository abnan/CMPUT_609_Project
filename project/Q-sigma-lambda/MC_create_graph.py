import csv
import matplotlib.pyplot as plt
import numpy as np

with open('MC_results.csv', 'r') as f:
    reader = csv.reader(f)
    result_list = list(reader)

results = np.array(result_list)
# colours = ['r--','b--','r-.','b-.','r-','b-','r:','b:','b--']
colours = ['r','g','b','y','p']

colour_index = 0

for i in range(0, results.shape[0]):
    plotx = np.arange(results.shape[1]-1)
    ploty = list(map(float, results[i,1:]))
    plt.plot(plotx, ploty, colours[colour_index], label = results[i,0])
    colour_index+=1

plt.xlabel('Episodes')
plt.ylabel('Average Return')
plt.legend()
plt.show()