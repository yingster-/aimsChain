import numpy as np

import cPickle as cp


class FIRE(object):
    def __init__(self, restart="fire.dat",
                 dt=0.02, maxmove=0.2, dtmax=1.0, Nmin=5, finc=1.1, fdec=0.5,
                 astart=0.1, fa=0.99, a=0.1):

        self.dt = dt
        self.Nsteps = 0
        self.maxmove = maxmove
        self.dtmax = dtmax
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = a
        self.restart = restart
        self.r0 = None

    def initialize(self):
        self.v = None

    def load(self):
        """load saved velocity and dt from file"""
        import os.path as path
        if path.isfile(self.restart):
            save = open(self.restart,'r')
            self.v, self.dt, self.a, self.Nsteps = cp.load(save)
            save.close()
    def dump(self):
        """dump necessary values for future reference"""
        save = open(self.restart, 'w')
        cp.dump((self.v, self.dt, self.a, self.Nsteps), save)
        save.close()
       
    def step(self,r,f):
        r = np.array(r)
        f = np.reshape(f,(-1,3))
        if self.v is None:
            self.v = np.zeros((len(f), 3))
        else:
            vf = np.vdot(self.v,f)
            if vf > 0.0:
                self.v = (1.0 - self.a) * self.v + self.a * f / np.sqrt(
                    np.vdot(f, f)) * np.sqrt(np.vdot(self.v, self.v))
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.Nsteps += 1
            else:
                self.v[:] *= 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.Nsteps = 0

        self.v += self.dt * f
        dr = self.dt * self.v
        maxdr = np.nanmax(np.sum(dr**2,1)**0.5)
        if maxdr > self.maxmove:
            dr = self.maxmove * dr / maxdr
        
        return r + dr.reshape(r.shape)
