import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

with open('test2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    lines = []
    first_line = True
    for row in csv_reader:
        if first_line:
            first_line = False
            continue
        lines.append(list([float(i) for i in row]))
    
    data = np.array(lines)
    
    tt = data[:, 0]
    x = data[:, 7]
    vx = data[:, 1]
    #x = data[:, 12]
    #vx = data[:, 6]
    
    # x_dot_raw = data[:, 16]
    
    forder = 2
    fcutoff = 10
    sample_rate = 20
    coeffs = scipy.signal.iirfilter(forder,
                                    fcutoff/sample_rate,
                                    btype='lowpass',
                                    output='ba',
                                    fs=sample_rate)
                     
    b = coeffs[0]               
    a = coeffs[1]
    
    x_dot_raw = np.concatenate((np.zeros(1), [(x[i] - x[i-1])/0.05 for i in range(1, len(x))]))
    
    x_dot_filt = []
        
    for i in range(len(x_dot_raw)):
        if i < forder:
            x_dot_filt.append(x_dot_raw[i])
            
        else:
            new_val = b[0] * x_dot_raw[i]
            
            for j in range(1, forder+1):
                new_val += b[j] * x_dot_raw[i-j]
                new_val -= a[j] * x_dot_filt[i-j]
            
            new_val = new_val / a[0]
            
            x_dot_filt.append(new_val)
        
    plt.plot(tt, vx)
    plt.plot(tt, x)
    plt.plot(tt, x_dot_raw)
    plt.plot(tt, x_dot_filt)
    plt.legend(['vx', 'x', 'x_dot_raw', 'x_dot_filt'])
    plt.ylim([-4, 4])
    plt.show()
