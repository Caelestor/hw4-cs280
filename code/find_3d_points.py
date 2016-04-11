import numpy as np

def find_3d_points( K1, K2, R, t, x1, x2 ):

    # K1, K2, are the (3 x 3) camera matrices for the two cameras C1, C2
    # R is the (3 x 3) rotation matrix describing the rotation from C2 to C1
    # t = (x_0, y_0, z_0) is the translation vector denoting the position of C1 in C2's coordinate
    # x1, x2 are (N x 2) arrays denoting the the position of reference points in each of the two images

    # Camera matrix for first camera with rotation and translation being the identity
    I0 = np.hstack( (np.identity(3), np.zeros((3,1)) ) ) 
    P1 = np.dot(K1, I0)

    # Camera matrix for the second matrix.
    Rt = np.hstack((R,t)) 
    P2 = np.dot(K2, Rt)

    N = len(x1) #Number of points

    X = np.zeros((N,4)) #Set of 3D points to be triangulatd in homogenous coords
    esq = 0 #Total error squared
    for i in range(N):

        # Solving for Af = 0
        A1 = np.outer([x1[i]], P1[2]) - P1[:2]
        A2 = np.outer([x2[i]], P2[2]) - P2[:2]
        A = np.vstack((A1,A2))
        U, S, V = np.linalg.svd( A )
        X[i] = V[-1]/V[-1,-1]

        # Adding up erorr
        e1 = np.linalg.norm(np.dot(A1, X[i])/np.dot(P1[2], X[i]))
        e2 = np.linalg.norm(np.dot(A2, X[i])/np.dot(P2[2], X[i]))
        esq = e1**2 + e2**2

    return X[:,:-1], np.sqrt(esq)/N

