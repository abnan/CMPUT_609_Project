import csv
import matplotlib.pyplot as plt
import numpy as np

x=[0,1,2]
y=x
plt.plot(x, y,'k-', label='λ = 0.7')
plt.plot(x, y,'k-.', label='λ = 0.3')
plt.plot(x, y,'k--', label='λ = 0.1')
plt.xlabel('Learning rate')
plt.ylabel('Average Return per Episode')
plt.legend()
plt.show()