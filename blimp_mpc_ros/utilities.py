from scipy.spatial.transform import Rotation as R
import numpy as np
from math import pi

def matrix_to_matlab(mat, name):

    print(name + ' = [', end='')
    for row in range(mat.shape[0]):
        if row > 0:
            print('\t', end='')
        for col in range(mat.shape[1]):
            print(mat[row][col], end=', ')
        
        if row < mat.shape[0] - 1:
            print(';')
    print('];')

def quat2euler(q):
    r = R.from_quat(q)
    return r.as_euler('XYZ', degrees=False)

def euler2quat(e):
    r = R.from_euler('xyz', e.T)
    return r.as_quat()
