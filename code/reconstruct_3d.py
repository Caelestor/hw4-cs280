# -*- coding: utf-8 -*-
"""
@author: richiou
"""

import os
import numpy as np
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt

def reconstruct_3d(name, plot=True):
    
    ## Load images, K matrices and matches
    data_dir = os.path.join('..', 'data', name)

    # images
    I1 = scipy.misc.imread(os.path.join(data_dir, "{}1.jpg".format(name)))
    I2 = scipy.misc.imread(os.path.join(data_dir, "{}2.jpg".format(name)))

    # K matrices
    K1 = np.array(scipy.io.loadmat(os.path.join(data_dir, "{}1_K.mat".format(name)))["K"], order='C')
    K2 = np.array(scipy.io.loadmat(os.path.join(data_dir, "{}2_K.mat".format(name)))["K"], order='C')

    # corresponding points
    # this is a N x 4 where:
    # matches[i,0:2] is a point in the first image
    # matches[i,2:4] is the corresponding point in the second image
    matches = np.loadtxt(os.path.join(data_dir, "{}_matches.txt".format(name)))

    # visualize matches (disable or enable this whenever you want)
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(np.concatenate([I1, I2], axis=1))
        ax.plot(matches[:, 0], matches[:, 1], 'r+')
        ax.plot(matches[:, 2] + I1.shape[1], matches[:, 3], 'r+')
        ax.plot(np.array([matches[:, 0], matches[:, 2] + I1.shape[1]]), matches[:, [1, 3]].T, 'r')

    # compute the fundamental matrix
    (F, res_err) = fundamental_matrix(matches)
    print('Residual in F = {}'.format(res_err))

    # compute the essential matrix
    E = np.dot(np.dot(K2.T, F), K1)
    

    # compute the rotation and translation matrices
    (R, t) = find_rotation_translation(E)
"""
    # Find R2 and t2 from R, t such that largest number of points lie in front
    # of the image planes of the two cameras
    P1 = np.dot(K1, np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1))

    # the number of points in front of the image planes for all combinations
    num_points = np.zeros((len(t), len(R)))

    # the reconstruction error for all combinations
    errs = np.empty((len(t), len(R)))

    for ti, t2 in enumerate(t):
        t2 = t[ti]
        for ri, R2 in enumerate(R):
            R2 = R[ri]
            P2 = np.dot(K2, np.concatenate([R2, t2[:, None]], axis=1))

            points_3d, errs[ti, ri] = find_3d_points()

            Z1 = points_3d[:, 2]
            Z2 = (np.dot(R2[2], points_3d.T) + t2[2]).T
            num_points[ti, ri] = np.sum((Z1 > 0) & (Z2 > 0))

    j = 0 # pick one out the best combinations
    (ti, ri) = np.nonzero(num_points == np.max(num_points))
    print('Reconstruction error = {}'.format(errs[ti[j], ri[j]]))

    t2 = t[ti[j]]
    R2 = R[ri[j]]
    P2 = np.dot(K2, np.concatenate([R2, t2[:, None]], axis=1))

    # compute the 3D points with the final P2
    points = find_3d_points()

    plot_3d()
    """

""" We find the fundamental matrix using the 8-Point algorithm """
    
def fundamental_matrix(matches):

    # Derive the normalization matrices
    image1 = matches[:,0:2]
    image2 = matches[:,2:4]
    m1 = np.mean(image1, axis=0)
    m2 = np.mean(image2, axis=0)       
    std1 = np.std(image1)
    std2 = np.std(image2)

    T1 = np.zeros((3, 3))
    T2 = np.zeros((3, 3))   
    T1[0][0] = 1.0/(std1*std1)
    T1[0][2] = -1*m1[0]
    T1[1] = (0, 1.0/(std1*std1), -1*m1[1])
    T1[2] = (0, 0, 1)
    T2[0] = (1.0/(std2*std2), 0, -1*m2[0])
    T2[1] = (0, 1.0/(std1*std1), -1*m2[1])
    T2[2] = (0, 0, 1)

    # Normalization

    onecoords = np.ones((len(matches),1))   
    image1 = np.append(image1, onecoords, axis=1)    
    image2 = np.append(image2, onecoords, axis=1) 

    image1 = np.dot(image1, T1.T)    
    image2 = np.dot(image1, T1.T)      

    """
    m0 = np.mean(matches, axis=0)
    m1 = np.mean(matches, axis=1)
    m2 = np.mean(matches, axis=2)
    m3 = np.mean(matches, axis=3)        

    v0 = np.std(matches, axis=0)
    v1 = np.std(matches, axis=1)
    v2 = np.std(matches, axis=2)
    v3 = np.std(matches, axis=3)  

    v0 = v0*v0
    v1 = v1*v1
    v2 = v2*v2
    v3 = v3*v3
    """
    
    points = np.random.choice(len(matches), 20)
    A = np.zeros((20, 9))
    print points
    n = 0

    # Create the 8-point matrix

    for i in points:
        p1 = image1[i]
        p2 = image2[i]
        A[n] = [p1[0]*p2[0], p1[1]*p2[0], p2[0], p1[0]*p2[1], p1[1]*p1[0], p2[1], p1[0], p1[1], 1]
        n = n + 1

    # Use SVD to solve Af = 0. Given A = USV^T, the last column of V is the solution.
    # http://www.cse.psu.edu/~rtc12/CSE486/lecture20_6pp.pdf 
     
    ua, sa, va = np.linalg.svd(A, full_matrices=True)    
    f = va.T[:,-1]
    f = np.reshape(f, (3, 3))

    # Use SVD to reduce F to rank 2 and Denormalize

    uf, sf, vf = np.linalg.svd(f, full_matrices=True)
    sf[2] = 0
    final = np.dot(np.dot(uf, np.diag(sf)),vf)
    final = np.dot(T2.T, final, T1)
    print final
    #return (final, 0)   
    
    # Compute Residuals
    residual = 0
    for point in matches:
        p1 = np.matrix((point[0], point[1], 1))
        p2 = np.matrix((point[2], point[3], 1))
        numerator = np.linalg.norm(np.dot(np.dot(p2, final), p1.T), ord=1)
        d1 = np.linalg.norm(np.dot(final, p1.T), ord=2)
        d2 = np.linalg.norm(np.dot(final, p2.T), ord=2)
        error = numerator*numerator*(1.0/(d1*d1) + 1.0/(d2*d2))
        residual = residual + error
    return (final, residual)

def find_rotation_translation(E):

    t = []
    R = []
    R_det = []
    #+-90 degree rotation matrices
    RCW90_t = np.transpose(np.matrix('0,-1,0;1,0,0;0,0,1'))
    RCCW90_t = np.transpose(np.matrix('0,1,0;-1,0,0;0,0,1'))

    #SVD
    U,Sig,V_t = np.linalg.svd(E)
    U_t = np.transpose(U)

    #t & R
    sign = [1,-1]
    rot = [RCW90_t,RCCW90_t]
    for s in sign:
        t.append(s*U[:,2])
        for r in rot:
            R_temp = s*np.transpose(np.dot(np.dot(U,r),V_t))
            R.append(R_temp)
            R_det.append(np.linalg.det(R_temp))
    #Only keep R's with determinant of 
    R_det = np.ndarray.tolist(np.array(R_det)-1)        
    for i in range(0,2):
        ind = R_det.index(max(R_det))
        R_det.pop(ind)
        R.pop(ind)
    return R, t


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
    
reconstruct_3d('house')
#reconstruct_3d('library')