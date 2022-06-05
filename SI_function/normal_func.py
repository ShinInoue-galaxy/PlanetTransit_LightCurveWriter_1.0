# normal function liblary
import scipy.optimize
import numpy as np

#--------3D plane fitting from
"""
https://gist.github.com/RustingSword/e22a11e1d391f2ab1f2c
"""
def fitPlane3D(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  #X
    G[:, 1] = XYZ[:, 1]  #Y
    Z = XYZ[:, 2]
    (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    a = normal[0]
    b = normal[1]
    c = normal[2]
    d = calc_d(a,b,c,XYZ)
    return a,b,c,d

def calc_d (a,b,c, XYZ):
    A = [a,b,c]
    d_sum = -np.dot(XYZ,A)
    return np.sum(d_sum)/len(d_sum)

#単なる二次元平面での距離の差
def dr(x_1, y_1, x_2, y_2):
    return ((x_1-x_2)**2+(y_1-y_2)**2)**(1/2)
