import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_3d(K1, K2, R, t, X):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = np.shape(X)[0]

    ax.scatter(X[:,0], X[:,1], X[:,2]) #Plot points

    # Camera 1
    ax.scatter(0,0,0,c='r') 
    f1 = K1[0,0]
    ax.quiver(0,0,0,0,0,1,length=f1,pivot='tail')

    # Camera 2
    C = -np.dot(R.T, t)
    ax.scatter(C[0],C[1],C[2],c='r') 
    ax.quiver(C[0],C[1],C[2],R[2,0],R[2,1],R[2,2],length=f1,pivot='tail')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
