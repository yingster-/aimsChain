"""
This module provide the parser for control file
control name is "chain.in"
"""
import os 
import string
class Control(object):
    """
    generic class for control
    """
    def __init__(self):
        #initial geometry
        self.ini = "ini.in"
        #final geometry
        self.fin = "fin.in"
        #number of images using
        self.nimage = 5
        #periodic interpolation
        self.periodic_interp = False
        #threshold for convergence
        self.thres = 0.1
        #threshold for climbing image convergence
        self.climb_thres = None
        #climbing image or not
        self.use_climb = False
        #climbing by interpolation
        self.climb_interp = True
        #climbing mode
        #1 for single node climb
        #2 for 3 node climb
        self.climb_mode = 1
        #command to run aims
        self.run_aims = "mpiexec -ppn 8 -n $NSLOTS ~/bin/aims.081912.scalapack.mpi.x"
        #using external starting geometry
        self.ext_geo = None
        #method, string or neb
        self.method = "string"
        #global optimizer
        self.global_opt = True
        #spring constant
        self.spring_k = 10.0
        #restart file
        self.aims_restart = None
        self.read()
    
    def read(self, filename = "chain.in"):
        if os.path.isfile(filename):
            control = open(filename, 'r')
            lines = control.readlines()
            control.close()
            for line in lines:
                inp = line.split()
                if inp == []:
                    continue
                elif inp[0][0] == '#':
                    continue
                elif inp[0] == "initial_file":
                    self.ini = inp[1]
                elif inp[0] == "final_file":
                    self.fin = inp[1]
                elif inp[0] == "n_images":
                    self.nimage = int(inp[1])
                elif inp[0] == "force_thres":
                    self.thres = float(inp[1])
                elif inp[0] == "climb_thres":
                    self.climb_thres = float(inp[1])
                elif inp[0] == "use_climb":
                    self.use_climb = parse_bool(inp[1])
                elif inp[0] == "climb_mode":
                    self.climb_mode = int(inp[1])
                elif inp[0] == "run_aims":
                    self.run_aims = string.join(inp[1:])
                elif inp[0] == "external_geometry":
                    self.ext_geo = inp[1]
                elif inp[0] == "method":
                    if inp[1] in ['NEB', 'neb', 'Neb']:
                        self.method = "neb"
                        print "Using NEB method for MEP finding\n"
                    else:
                        self.method = 'string'
                        print "Using string method for MEP finding\n"
                elif inp[0] == "global_optimizer":
                    self.global_opt = parse_bool(inp[1])
                elif inp[0] == "interpolated_climb":
                    self.climb_interp = parse_bool(inp[1])
                elif inp[0] == "neb_spring_constant":
                    self.spring_k = float(inp[1])
                elif inp[0] == "periodic_interpolation":
                    self.periodic_interp = parse_bool(inp[1])
                elif inp[0] == "aims_restart":
                    self.aims_restart = str(inp[1])

        #assign climbing thres if it's not set
        if self.climb_thres == None:
            self.climb_thres = self.thres



def parse_bool(string):

    if string in ['true','True','.true.', 'Y', 'Yes', 'YES'
                  'TRUE', 'y', 'yes', '1', 't']:
        return True
    else:
        return False
