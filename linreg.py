import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def linreg(A, b):
    if len(A.shape) == 1:
        return np.dot((1./np.dot(A.transpose(),A))*A.transpose(),b)
    return np.dot(np.dot(inv(np.dot(A.transpose(),A)),A.transpose()),b)

def expand_data(A,n):
    result = np.zeros((A.shape[0],n))
    for i in range(0,n):
        result[:,i] = np.power(A,i)
    return result

if __name__ == "__main__":
    X = np.random.rand(100)*8 -2
    Y = np.random.rand(100)* -0.6*X - 0.4*X**2 + 0.1*X**3
    w = linreg(expand_data(X,5),Y)
    p = np.arange(-2,5,0.01)
    p_new = expand_data(p,5)
    plt.plot(p, np.sum(w*p_new,axis=1))
    plt.plot(X, Y, "bo")
    plt.show()
