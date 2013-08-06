"""
The growing string path
"""
import numpy as np
from aimsChain.utility import vmag, vunit, vproj
from aimsChain.gs_path import GrowingStringPath
from aimsChain.node import Node
from aimsChain.optimizer.optimize import FDOptimize
import cPickle as cp
class PulledGrowingStringPath(GrowingStringPath):

    @property
    def lowest_moving(self):
        nspace = self.lower.param/self.linsep-1
        nspace = max(0,nspace)
        nspace = self.lower_end - int(round(nspace))
        return max(1,nspace)

    @property
    def highest_moving(self):
        nspace = (1.0-self.upper.param)/self.linsep-1
        nspace = max(0,nspace)
        nspace = self.upper_end + int(round(nspace))
        return min(nspace, len(self.nodes)-2)



    def __init__(self, 
             nodes = [],
             control = None):
        self.lower_end = 0
        super(GrowingStringPath,self).__init__(nodes,control)


    def __reorgnize_nodes(self):
        from aimsChain.interpolate import spline_pos
        positions = []
        new_t = np.linspace(0,1,self.control.gs_nimage+2)
        for i in self.get_valid_index():
            node = self.nodes[i]
            positions.append(node.positions)
        
        new_pos = spline_pos(np.array(positions), new_t)

        for i in range(1,len(new_t)-1):
            self.nodes[i].param = new_t[i]
            self.nodes[i].positions = new_pos[i]
            self.nodes[i].update_dir()



    def not_converged(self):
        if ((self.upper.param - self.lower.param)
            <= self.linsep):
            self.__reorgnize_nodes()
            return False
        else:
            return True


    def add_lower(self):
        from aimsChain.interpolate import spline_pos,get_t

        #if path is grown do nothing
        if not self.not_converged():
            self.__reorgnize_nodes()

        #get new coordinate for lower_node
        moving_index = self.get_valid_index()
        self.lower.param += self.linsep
        positions = []
        old_pos = []
        for i,node in enumerate(self.nodes):
            if i in moving_index:
                positions.append(node.positions)
                if i <= self.lower_end:
                    old_pos.append(node.positions)
        self.lower.positions = spline_pos(positions,[self.lower.param])[0]
        old_pos.append(self.lower.positions)

        old_t = get_t(old_pos)*self.lower.param

        nodes = []
        moving_index = self.get_valid_index()
        for i,node in enumerate(self.nodes):
            if (i in moving_index) and (i <= self.lower_end):
                nodes.append(node) 
        new_t = np.linspace(0,self.lower.param,len(nodes))
        new_pos = spline_pos(old_pos,new_t,old_t)[1:-1]
        new_t = new_t[1:-1]
        
        for i,node in enumerate(nodes[1:-1]):
            node.param = new_t[i]
            node.positions = new_pos[i]

        return True

        
    def add_upper(self):    
        from aimsChain.interpolate import spline_pos,get_t
        
        if not self.not_converged():
            self.__reorgnize_nodes()


        moving_index = self.get_valid_index()
        self.upper.param -= self.linsep 

        positions = []
        old_pos = []
        for i,node in enumerate(self.nodes):
            if i in moving_index:
                positions.append(node.positions)
                if i >= self.upper_end:
                    old_pos.append(node.positions)
        self.upper.positions = spline_pos(positions,[self.upper.param])[0]
        
        old_pos.insert(0,self.upper.positions)

        old_t = get_t(old_pos)*(1-self.upper.param) + self.upper.param

        

        nodes = []
        moving_index = self.get_valid_index()
        for i,node in enumerate(self.nodes):
            if (i in moving_index) and (i >= self.upper_end):
                nodes.append(node) 
        new_t = np.linspace(self.upper.param,1,len(nodes))
        new_pos = spline_pos(old_pos,new_t,old_t)[1:-1]
        new_t = new_t[1:-1]
        
        for i,node in enumerate(nodes[1:-1]):
            node.param = new_t[i]
            node.positions = new_pos[i]


        return True




    def lower_tangent(self):
        from aimsChain.interpolate import spline_pos
        if (self.upper.param - self.lower.param) <= self.linsep*1.2:
            return self.lower.get_tangent()
        positions = []
        param = []
        moving = self.get_valid_index()
        for i,node in enumerate(self.nodes):
            if i in moving:
                positions.append(node.positions)
                param.append(node.param)

        loc = param.index(self.lower.param)
        derv = spline_pos(positions, param, param, 3, 1)[loc]
        tangent = derv
        
        return vunit(tangent)

    def upper_tangent(self):
        if (self.upper.param - self.lower.param) <= self.linsep*1.2:
            return self.upper.get_tangent()
        from aimsChain.interpolate import spline_pos

        positions = []
        param = []
        moving = self.get_valid_index()
        for i,node in enumerate(self.nodes):
            if i in moving:
                positions.append(node.positions)
                param.append(node.param)

        loc = param.index(self.lower.param)
          
        derv = spline_pos(positions, param, param, 3, 1)[loc]
        tangent = derv
        
        return vunit(tangent)



    def move_nodes(self):
#        print self.params
        forces = None
        if not self.not_converged:
            forces = self.move_nodes_combined()
        else:
            forces = self.move_nodes_separated()

        return forces

    def move_nodes_separated(self):
        from aimsChain.interpolate import spline_pos, get_total_length, get_t
        import os
        positions = []
        forces = []
        new_pos = []
        new_pos2= []
        total_length = 0
        low_force = self.lower_force()
        high_force = self.upper_force()
        moving = self.get_valid_index()

        #extract forces and positions from the nodes
        for i,node in enumerate(self.nodes):
            positions.append(node.positions)
            if not (i in moving):
                #get zero forces if it's dummy node
                forces.append(np.zeros(np.shape(node.positions)))
            elif node is self.upper:
                forces.append(self.upper_force())
                high_force = self.upper_force()
            elif node is self.lower:
                forces.append(self.lower_force())
                low_force = self.lower_force()
            else:
                forces.append(node.normal_forces)

        forces = np.array(forces)
        positions = np.array(positions)
        
        save = os.path.join(self.nodes[0].dir_pre, "pgs.opt")
        new_pos,opt = self.g_opt(
            self.control.gs_optimizer,
            positions,
            forces,
            save)

        moving_nodes = []
        moving_param = []
        lower_end = 0
        for i,node in enumerate(self.nodes):
            if i in moving:
                new_pos2.append(new_pos[i])
                moving_nodes.append(node)
                moving_param.append(node.param)
                if i == self.lower_end:
                    lower_end = moving_nodes.index(node)
        upper_end = lower_end + 1
        new_pos = new_pos2
        new_pos2 = []
                    
        lower_length = get_total_length(new_pos[:lower_end+1])
        upper_length = get_total_length(new_pos[upper_end:])
        total_length = get_total_length(new_pos)


        #decide whether to reparm locally or globally
        lower_ideal = self.lower.param*total_length
        upper_ideal = total_length - self.upper.param*total_length

        if (abs(lower_length-lower_ideal) > 0.1 * lower_ideal or 
            abs(upper_length - upper_ideal) > 0.1 * upper_ideal):
            if not (isinstance(opt, FDOptimize) and opt.finite_diff):
                new_pos2 = spline_pos(np.array(new_pos), moving_param)
            else:
                new_pos2 = new_pos

            
        else:

            temp_pos = new_pos[:lower_end+1]
            new_t = np.linspace(0.0,1.0,len(temp_pos))
            
            if not (isinstance(opt, FDOptimize) and opt.finite_diff):
                temp_pos = spline_pos(temp_pos, new_t)
            new_pos2.extend(temp_pos)

            temp_pos = new_pos[upper_end:]
            new_t = np.linspace(0.0,1.0,len(temp_pos))
            
            if not (isinstance(opt, FDOptimize) and opt.finite_diff):
                temp_pos = spline_pos(temp_pos, new_t)
                
            new_pos2.extend(temp_pos)


        for i,position in enumerate(new_pos2):
            moving_nodes[i].positions = position


        high_force = np.reshape(high_force, (-1,3))
        high_force = np.sum(high_force**2,1)**0.5

        low_force = np.reshape(low_force, (-1,3))
        low_force = np.sum(low_force**2,1)**0.5

        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5
        return np.nanmax(forces),np.nanmax(low_force),np.nanmax(high_force)


    
    def move_nodes_combined(self):
        from aimsChain.interpolate import spline_pos, get_total_length
        import os
        
        positions = []
        forces = []
        new_t = self.params
        new_pos = []
        low_force = self.lower_force()
        high_force = self.upper_force()

        for i,node in enumerate(self.nodes):
            positions.append(node.positions)
            forces.append(node.normal_forces)

        forces = np.array(forces)
        positions = np.array(positions)
        
        
        save = os.path.join(self.nodes[0].dir_pre, "pgs.opt")
        new_pos,opt = self.g_opt(
            self.control.gs_optimizer,
            positions,
            forces,
            save)

        if not (isinstance(opt, FDOptimize) and opt.finite_diff):
#            print new_pos
#            print new_t
#            print "----"
            new_pos = spline_pos(new_pos, new_t)

        for i,position in enumerate(new_pos):
            self.nodes[i].positions = position



        
        high_force = np.reshape(high_force, (-1,3))
        high_force = np.sum(high_force**2,1)**0.5

        low_force = np.reshape(low_force, (-1,3))
        low_force = np.sum(low_force**2,1)**0.5

        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5
        return np.nanmax(forces),np.nanmax(low_force),np.nanmax(high_force)



    def write_path(self, filename="path.dat"):
        """
        Write the current path into a file
        Will contain:
        current runs
        each node's param, dir, and whether they are fixed or not
        This was preferred over pickling for the sake of easy debugging
        May change to pickling in the production level
        """
        data = {}
        data["runs"] = self.runs
        params = []
        dirs = []
        fix = []
        climb = []
        for node in self.nodes:
            params.append(node.param)
            dirs.append(node.dir)
            fix.append(node.fixed)
            climb.append(node.climb)

        data["param"] = params
        data["dirs"] = dirs
        data["fix"] = fix
        data["climb"] = climb
        data["lower"] = self.lower_end

        save = open(filename,'w')
        cp.dump(data,save)
        save.close()

    def read_path(self, filename="path.dat"):
        """
        Read the file into current path
        """
        nodes = []
        save = open(filename, 'r')

        data = cp.load(save)
        save.close()

        self.runs = data["runs"]
        params = data["param"]
        dirs = data["dirs"]
        fix = data["fix"]
        climb = data["climb"]
        self.lower_end = data["lower"]


        for i,param in enumerate(params):
            tmp_node = Node(param, path=self)
            tmp_node.dir = dirs[i]
            tmp_node.fixed = fix[i]
            tmp_node.climb = climb[i]
            
            nodes.append(tmp_node)
    
        self.nodes = nodes



    def load_nodes(self):
        """
        read geometry.in and aims output, 
        will find coordinate from geometry, 
        and forces/energy from aims output
        if aims output is not present, only
        geometry will be read
        """
        import os
        from aimsChain.aimsio import read_aims_output, read_aims
        index = self.get_valid_index()
        for i,node in enumerate(self.nodes):
            if i in index:
                dir = os.path.join(node.dir_pre, node.dir)
                geo_path = os.path.join(dir, "geometry.in")
                output_path = os.path.join(dir, node.dir+".out")
                atoms = read_aims(geo_path)
                node.geometry = atoms
                if os.path.isfile(output_path):
                    ener,forces = read_aims_output(output_path)
                    node.ener = ener
                    node.forces = forces

    def write_node(self, control_file="control.in", fixed = False):
        """
        create a new folder for each node,ignore fixed
        including geometry and control
        name is derived from node.dir
        """
        import os
        import shutil
        from aimsChain.aimsio import write_aims
        path = []
        index = self.get_valid_index()
        for i,node in enumerate(self.nodes):
            if i in index:
                node.update_dir(fixed)
                dir=node.write_node(fixed, control_file)
                if dir != None:
                    path.append(dir)
        return path

    def get_valid_index(self, lowest=None, highest=None):
        if lowest == None:
            lowest = self.lowest_moving
        if highest == None:
            highest = self.highest_moving
        index = [0]
        index.extend(range(self.lowest_moving,self.highest_moving+1))
        index.append(len(self.nodes)-1)
        return index


    def init(self):
        """
        interpolate/resample the current path
        """
        from aimsChain.interpolate import spline_pos
        import copy
        
        n = self.control.gs_nimage
        nodes = []

        #insert two nodes first
        self.insert_node(self.linsep)
        self.insert_node(1-self.linsep)

        ininode = self.nodes[0]
        lower_node = self.nodes[1]
        upper_node = self.nodes[2]
        finnode = self.nodes[3]

        #add dummy nodes
        lower_n = int((n-2)/2)
        upper_n = int(n-lower_n)

        nodes.append(ininode)        
        
        for _ in range(lower_n):
            tmp_node = Node(param = 0.0,
                            geometry = copy.deepcopy(ininode.geometry),
                            path = self)
            nodes.append(tmp_node)

        nodes.append(lower_node)
        nodes.append(upper_node)
        
        for _ in range(lower_n):
            tmp_node = Node(param = 1.0,
                            geometry = copy.deepcopy(finnode.geometry),
                            path = self)
            nodes.append(tmp_node)

        nodes.append(finnode)

        self.nodes = nodes

        #set necessary indexes
        self.lower_end = self.nodes.index(lower_node)
