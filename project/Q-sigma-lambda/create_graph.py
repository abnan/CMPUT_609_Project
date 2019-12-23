import csv
import matplotlib.pyplot as plt
import numpy as np

with open('results.csv', 'r') as f:
    reader = csv.reader(f)
    result_list = list(reader)

results = np.array(result_list)
colours = ['r--','b--','r-.','b-.','r-','b-','r:','b:','b--']
# colours = ['r','g','b','y','p']

colour_index = 0
alphas = results[0, 1:]

for i in range(1, results.shape[0]):
    plotx = alphas
    ploty = list(map(float, results[i, 1:]))
    plt.plot(plotx, ploty, colours[colour_index], label = results[i,0])
    colour_index+=1

plt.xlabel('Learning rate')
plt.ylabel('Return')
plt.legend()
plt.show()