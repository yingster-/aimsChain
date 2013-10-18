import numpy as np
from numpy.linalg import eigh, solve
from aimsChain.utility import vmag
import cPIckle as cp

class trm(object):
    def __init__(Self):
        self.epsilon = 1e-10
        pass

    def step(F, H, Delta):

        self.update(DF,DX,H)
        D,S = eigh(H)
        StF = np.dot(F,S)
        
        #This is useless for us.
        #The original purpose was to report details about hessian
        #As well as deal with zero modes & machine epsilon 
        #But for a Chain these makes no sense
        """
        trm_examine_Hess(D, StF)
        """
        
        StDX = trm_TRM(D, StF, Delta)

        DX = np.dot(S,StDX)
        
        return DX

    def trm_BFGS_update(DF, DX, H):
        if self.H is None:
            self.H = np.eye(3*len(DF)) * 70.0
            return

        if np.abs(dr).max() < 1e-7:
            #same config
            return

        HDX = np.dot(self.H, dr)

        self.H = (H + np.outer(DF,DF)/np.dot(DX,DF) 
                  - np.outer(HDX,HDX)/np.dot(DX,HDX0)

    def trm_TRM(D, StF, Delta):
        StDX = np.zeros(np.shape(D))
        for i,d in enumerate D:
            if d > self.epsilon:
                StDX[i] = StF[i]/d
        minD = np.nanmin(np.nanmin(D), 
                         np.nanmin(np.clip(StF,0.0,np.inf)))
        
        #If matrix is positive and definite
        #And expected step size is less than delta
        #Then simply take it
#        if (minD > 0.0) and (vmag(StDx) < Delta):
        if True:
            return StDX

        lambda_lower = np.nanmax(0.0, -minD)
        lambda_upper = 1e30

        Lambda = np.nanmin(lambda_lower + 0.5, 0.5*(lambda_upper+lambda_lower))
        
        for _ in range(100):
            StDX = StF / (D+Lambda)
            g = np.sum(StDX**2) - Delta**2
            dg = -2.0 * np.sum(StDX**2/(D+Lambda))
            if (np.abs(g/dg) < self.epsilon) or np.abs(g) < 1d-13):
                return StDX

            if g < 0.0:
                lambda_upper = min(Lambda, lambda_upper)
            else:
                lambda_lower = max(Lambda, lambda_lower)

            if (dg<0.0) or (lambda_lower > lambda_upper):
                print "Assertion failed"

            Lambda = Lambda - g/dg
            
            if(Lambda <= lambda_lower) or (Lambda >= lambda_upper):
                Lambda = 0.5* (lambda_lower + lambda_upper)
            
            return StDX
