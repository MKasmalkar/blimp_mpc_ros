import csv
from utilities import *
import matplotlib.pyplot as plt
import numpy as np

with open('quat_data.csv', mode='r') as file:
        csvfile = csv.reader(file)
        lines = []
        first_line = True
        for line in csvfile:
                if first_line:
                        first_line = False
                        continue
                lines.append([float(i) for i in line])
                
        data_np = np.array(lines)
                    
        y = data_np[:, 1]
        x = data_np[:, 2]
        z = data_np[:, 3]
        w = data_np[:, 4]

        data = []

        for i in range(len(x)):
                q = quat2euler(np.array([x[i], y[i], z[i], w[i]]))
                data.append(q)
                
        data = np.array(data)
        print(data.shape)
                
        plt.plot(data)
        plt.legend(['phi', 'theta', 'psi'])
        plt.show()
