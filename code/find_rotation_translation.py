import numpy as np
#import pudb; pu.db
def find_rotation_translation():
	#E = np.matrix('-0.00310695,-0.0025646,2.96584;-0.028094,-0.00771621,56.3813;13.1905,-29.2007,-9999.79')
	t = []
	R = []
	R_det = []
	#+-90 degree rotation
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
				
	#Only keep R's with determinant of 1           
    for i in range(0,2):
        ind = R_det.index(max(np.absolute(R_det - 1)))
        R_det.pop(ind)
        R.pop(ind)
    return R, t
