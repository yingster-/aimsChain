import numpy as np
from aimsChain.utility import vunit
from aimsChain.optimizer.optimize import Optimize

import cPickle as cp

class CG(Optimize):
    def __init__(self, restart="restart", maxstep = 0.04, sec_step = 0.001, safe_step = 0.04):
        self.restart = restart
        self.maxstep = maxstep
        self.sec_step = sec_step
        self.safe_step = safe_step

            
    
    def initialize(self):
        self.kter = 0
        self.prev_alpha = []
        self.d = None
        self.r = None
        self.r_prev = None
        self.N = None
        self.dr = None
        self.x0 = None
        self.f_prime = None
        self.sec_dis = None
        self.prev_step = 0.001
        self.finite_diff = False
        

    def load(self):
        """
        load saved infos
        """
        import os.path as path
        if path.isfile(self.restart):
            hess = open(self.restart, 'r')
            try:
                (self.kter,
                 self.prev_alpha,
                 self.d,
                 self.r,
                 self.r_prev,
                 self.N,
                 self.dr,
                 self.x0,
                 self.f_prime,
                 self.sec_dis,
                 self.finite_diff) = cp.load(hess)
            except:
                self.initialize()
            hess.close()

    def dump(self):
        """
        dump necessary values for future reference
        """
        hess = open(self.restart, 'w')
        cp.dump((self.kter,
                 self.prev_alpha,
                 self.d,
                 self.r,
                 self.r_prev,
                 self.N,
                 self.dr,
                 self.x0,
                 self.f_prime,
                 self.sec_dis,
                 self.finite_diff),
                hess)
        hess.close()

    def step(self, r, f):
        """
        Take a single step
        """
        force = np.array(f).flatten()
        if np.abs(f).max() < 1e-7:
            return r #Too small to go on

        if not self.N: #degree of freedom
            self.N = int(len(f)*0.5+1) #threshold set for restarting
        if self.d == None: #initial direction
            self.d = force
        if self.r_prev == None:
            self.r_prev = force

        if not self.finite_diff:
            self.x0 = np.array(r)
            self.r = force
            self.f_prime = -1 * force
            self.finite_diff = True
            
            self.sec_dis = min(
                self.determine_alpha(self.d, self.sec_step),
                self.determine_alpha(self.d, self.prev_step*0.1))

            dx = self.sec_dis*self.d
            return self.x0 + np.reshape(dx, np.shape(self.x0))

        self.finite_diff = False
        
        #####Finite difference, secant method
        finite_f = -1 * force
        
        #1step newton method
        
        fp1 = np.sum(np.dot(self.f_prime,vunit(self.d)))
        fp2 = np.sum(np.dot(finite_f,vunit(self.d)))
        curv = (fp2-fp1)/self.sec_dis
        alpha = 0.5*(fp1+fp2)/curv
#        print curv
        """
        #secant method
        alpha = -1 * self.sec_dis *  (np.dot(self.f_prime, self.d)/
                     (np.dot(finite_f, self.d) -
                      np.dot(self.f_prime, self.d)))
        """
        #####step the coordinate
        #in case of convex and etc, alpha may be negative
        #in this case we take the minimum of:
        #                  avg over past 5 alpha
        #                  time for safe step
        
        if alpha <= 0:
            if len(self.prev_alpha) <= 2:
                alpha = self.determine_alpha(self.d, self.safe_step)
            else:
                alpha = min(
                    np.average(self.prev_alpha[-5:]),
                    self.determine_alpha(self.d, self.safe_step)
                    )
        
        dx = alpha * self.d
        #rescale dx so it
        dx = dx.reshape(-1,3)
        dx,factor = self.determine_step(dx)
        self.prev_step = np.max((dx**2).sum(1)**0.5)
        dx = np.reshape(dx, np.shape(self.x0))
        self.prev_alpha.append(factor*alpha)
        #####
        
        ####update CG direction
        beta = max(0, 
                   (np.dot(self.r,(self.r-self.r_prev))/
                    (np.dot(self.r_prev,self.r_prev))))
        if self.kter >= self.N: # reset every N iterations
            beta = 0
            self.kter = 0

        self.d = self.r + beta*self.d
        #####
        self.kter += 1
        return self.x0 + dx

    def determine_alpha(self, f, step):
        """
        return the "time" required to make
        the given force move at most length=step
        """
        f = f.reshape(-1,3)
        f = (f**2).sum(1)**0.5
        maxlength = np.max(f)
        return step/maxlength


