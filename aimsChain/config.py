"""
This module provide the parser for control file
control name is "chain.in"
"""
import os 
import string
class control(object):
    """
    generic class for control
    """
    def __init__(self):
        self.ini = "ini.in"
        self.fin = "fin.in"
        self.nimage = 5
        self.thres = 0.1
        self.climb_thres = None
        self.use_climb = False
        self.run_aims = "mpiexec -ppn 8 -n $NSLOTS ~/bin/aims.081912.scalapack.mpi.x"
        self.ext_geo = None
        self.method = "string"
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
        #assign climbing thres if it's not set
        if self.climb_thres == None:
            self.climb_thres = self.thres



def parse_bool(string):

    if string in ['true','True','.true.', 'Y', 'Yes', 'YES'
                  'TRUE', 'y', 'yes', '1', 't']:
        return True
    else:
        return False
