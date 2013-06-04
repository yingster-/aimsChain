import numpy as np
from numpy.linalg import eigh, solve

import cPickle as cp

class dampedBFGS(object):
    def __init__(self, restart="hess", maxstep=0.04, 
                 alpha = 70):
        """BFGS optimizer.while force > 0.01:

        Parameters:

        restart: string
            Pickle file used to store hessian matrix. 
            load() will attempt to load existing Hessian from it.
            dump() will dump the current Hessian to it. 
        Maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.04 A).
        """
        
        if maxstep is not None:
            if maxstep > 1.0:
                raise ValueError('You are using a much too large value for ' +
                                 'the maximum step size: %.1f A' % maxstep)
            self.maxstep = maxstep
        self.restart = restart
        self.alpha = alpha

    def initialize(self):
        self.H = None
        self.r0 = None
        self.f0 = None

        

    def load(self):
        """
        load saved hessian, positions, and forces
        """
        import os.path as path
        if path.isfile(self.restart):
            hess = open(self.restart, 'r')
            self.H, self.r0, self.f0 = cp.load(hess)
            hess.close()

    def dump(self):
        """
        dump necessary values for future reference
        """
        hess = open(self.restart, 'w')
        cp.dump((self.H, self.r0, self.f0),
                hess)
        hess.close()

    def step(self, r, f):
        """
        Tkae a single step
        """
        r = np.array(r)
        f = np.array(f).flatten()

        self.update(r, f)

        dr = self.H.dot(f)

        dr = dr.reshape(-1,3)
        dr = self.determine_step(dr)
        dr = np.reshape(dr, np.shape(r))

        self.r0 = r.flatten()
        self.f0 = f.flatten()


        return r+dr

    def determine_step(self, dr):
        """Determine step to take according to maxstep
        
        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        
        steplengths = (dr**2).sum(1)**0.5
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            dr *= self.maxstep / maxsteplength
        
        return dr

    def update(self, r, f):

        r = r.flatten()
        f = f.flatten()
        if self.H is None:
            self.H = np.eye(len(r)) * (1.0/self.alpha)
            self.r0 = r
            self.f0 = f
            return

        r0 = self.r0
        f0 = self.f0

        sk = r.reshape(-1) - r0.reshape(-1)

        if np.abs(sk).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        #force is negative!
        
        yk = f0.reshape(-1) - f.reshape(-1)
        rhok=np.dot(yk,sk)
        
        #print sk
        #print yk
        print "rhok  " + str(rhok)
        theta = 1
        thres = 0.2*np.dot(sk,self.H).dot(sk)
        
        if rhok < thres:
            theta = (4*thres)/(5*thres-rhok)
#            print "damped!"
#            self.H = np.eye(len(r))*(0.1/self.alpha)
            return
        #yk =  theta * yk + (1 -theta)*np.dot(sk,self.H)
        #BFGS update formula
        I= np.eye(len(r))
        try:
            rhok = 1.0/(np.dot(yk,sk))
        except ZeroDivisionError:
            rhok = 1000.0
        if np.isinf(rhok) or float(rhok) >=1000:
            rhok = 1000.0
        A1 = I - sk[:,np.newaxis] * yk[np.newaxis,:] * rhok
        A2 = I - yk[:,np.newaxis] * sk[np.newaxis,:] * rhok
        
        self.H = (np.dot(A1,np.dot(self.H,A2)) 
                  + rhok * np.outer(sk,sk))
        