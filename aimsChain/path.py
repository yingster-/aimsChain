"""
This module defines the Path
Path is a collection of nodes
Path has all the inter-node functions
"""
import numpy as np
from aimsChain.utility import vmag, vunit, vproj
from aimsChain.config import Control
from aimsChain.node import Node


class Path(object):
    """
    Class for a path
    Parameters:
    nodes: list of nodes that the path contains
    """
    def __init__(self, 
                 nodes = [],
                 control = None):
        self.__nodes = []
        for node in nodes:
            node.path=self
            self.__nodes.append(node)
        self.__runs = 0
        self.__control = control
        if self.__control == None:
            self.__control = Control()

    @property
    def control(self):
        return self.__control
    @propertys
    def nodes(self):
        """get a list of all nodes in the path"""
        return self.__nodes
    @nodes.setter
    def nodes(self,node):
        """
        take either a single node or a list of nodes
        in case of a single node, insert it into self.nodes according to param
        in case of a list of nodes, set it as self.nodes
        """
        if isinstance(node, Node):
            node.path = self
            self.__nodes.append(node)
            self.__nodes.sort(key=lambda x: x.param)
        elif isinstance(node, list):
            for n in node:
                n.path = self
            self.__nodes = node
    @property
    def runs(self):
        return self.__runs
    @runs.setter
    def runs(self, value):
        self.__runs = int(value)
    @property
    def periodic(self):
        return self.nodes[0].geometry.lattice != None
    @property
    def lattice_vector(self):
        return self.nodes[0].geometry.lattice

    def add_runs(self):
        self.__runs += 1
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
        for node in self.nodes:
            dir = os.path.join(node.dir_pre, node.dir)
            geo_path = os.path.join(dir, "geometry.in")
            output_path = os.path.join(dir, node.dir+".out")
            atoms = read_aims(geo_path)
            node.geometry = atoms
            if os.path.isfile(output_path):
                ener,forces = read_aims_output(output_path)
                node.ener = ener
                node.forces = forces

    def write_node(self, control_file="control.in"):
        """
        create a new folder for each node,ignore fixed
        including geometry and control
        name is derived from node.dir
        """
        import os
        import shutil
        from aimsChain.aimsio import write_aims
        path = []
        for node in self.nodes:
            node.update_dir()
            dir=node.write_node(control_file=control_file)
            if dir != None:
                path.append(dir)
        return path
    
    def write_all_node(self, control_file = "control.in"):
        """
        create a new folder for each node, even fixed
        including geometry and control
        name is derived from node.dir
        """
        import os
        import shutil
        from aimsChain.aimsio import write_aims
        path = []
        for node in self.nodes:
            node.update_dir(True)
            path.append(node.write_node(True, control_file))
        return path
    
    def get_paths(self, return_fixed = True):
        """
        return a list of all paths in the current nodes
        will not return fixed nodes if return_fixed is false
        """
        import os
        path = []
        for node in self.nodes:
            if (not node.fixed) or return_fixed:
                path.append(os.path.join(node.dir_pre, node.dir))
        return path


    def insert_node(self, param):
        """
        insert a new node at the specified param
        """
        from aimsChain.interpolate import spline_pos
        import copy
        
        k = 3
        if self.n_nodes() <= 3:
            k = self.n_nodes() - 1
        
        positions = []
        old_t = []
        for node in self.nodes:
            old_t.append(node.param)
            positions.append(node.positions)
        
        new_t = list(old_t)
        new_t.append(float(param))
        new_t.sort()
        ind = new_t.index(param)

        positions = spline_pos(np.array(positions), new_t, old_t, k=k)

        new_node = Node(param = param,
                        geometry = copy.deepcopy(self.nodes[0].geometry),
                        path=self)
        new_node.positions = positions[ind]
        new_node.update_dir()

        self.nodes = new_node
        return new_node


    def move_nodes(self):
        """
        move the node according to forces
        only euler method for now
        """
        from aimsChain.optimizer.euler import EULER
        from aimsChain.interpolate import spline_pos
        positions = []
        forces = []
        new_t = []
        diff_pos = None
        t_step = 0.03
        for node in self.nodes:
            #list all the new params
            new_t.append(node.param)
            #add pos and force to the list
            #only if it is not new (0 been default value)
            if (np.sum(node.positions) != 0):
                positions.append(node.positions)
                forces.append(node.forces)

        forces = np.array(forces)
        positions = np.array(positions)
        opt = EULER(t_step)
        new_pos = opt.step(positions, 
                        forces)
        new_pos = spline_pos(new_pos, new_t)

        #check difference only if shape remain unchanged
        #safeguards against dynamic interpolation
        if positions.shape == new_pos.shape:
            diff_pos = new_pos - positions
            diff_pos = np.reshape(diff_pos, (-1,3))
            diff_pos = np.sum(diff_pos**2,1)**0.5
    
        for i,position in enumerate(new_pos):
            self.nodes[i].positions = position

        if diff_pos != None:
            diff_pos = np.nanmax(diff_pos)
            return diff_pos/t_step
        else:
            return 1.0


    def find_climb(self,returning = False):
        """
        turn on the climb flags in the list of nodes
        """
        from aimsChain.interpolate import arb_interp, spline_pos
        energy = []
        climb_nodes = []
        positions = []
        old_t = []
        target_node = None
        climb_mode = self.control.climb_mode

        for node in self.nodes:
            energy.append(node.ener)
            old_t.append(node.param)
            positions.append(node.positions)
            node.climb = False
            if climb_mode != 3:
                node.fixed = True

        #just go for the highest energy if we are not interpolating
        #or it's returning seek
        if (self.control.climb_interp == False) or returning:
            ind = energy.index(np.nanmax(energy[1:-1]))
            self.nodes[ind].fixed = False
            self.nodes[ind].climb = True
            target_node = self.nodes[ind]
        else:
            new_t = np.linspace(0,1,1001)
            energy_interp = arb_interp(energy, new_t, old_t)
            ind = np.where(energy_interp == np.nanmax(energy_interp[1:-1]))[0][0]
            #we want to see if the highest energy point is near a existing node
            change_t = old_t - new_t[ind]
            mint_ind = np.where(change_t == np.nanmin(np.absolute(change_t)))[0][0]
            
            if(change_t[mint_ind] <= 0):
                t_thres = old_t[mint_ind+1] - old_t[mint_ind]
            else:
                t_thres = old_t[mint_ind] - old_t[mint_ind-1]
            t_thres = t_thres/4.0
            #if it's not within the center 1/2 of the path, then we say it's too close
            #to a existing node, and we use that instead
            if np.nanmin(change_t) < t_thres:
                self.nodes[mint_ind].fixed = False
                self.nodes[mint_ind].climb = True
                target_node = self.nodes[mint_ind]
            else:
                new_t = np.array(new_t[ind])
                new_node = self.insert_node(new_t)                
                new_node.climb = True
                new_node.fixed = False
                new_node.ener = np.nanmax(energy_interp[1:-1])
                target_node = new_node
                self.nodes = new_node
        if climb_mode == 2:
            target_node.prev.fixed = False
            target_node.next.fixed = False
            


    def move_climb(self):
        """
        move the climbing nodes using BFGS
        """
        from aimsChain.interpolate import spline_pos, get_t
        import os
        moving_nodes = []
        forces = []
        positions = []
        climb_mode = self.control.climb_mode
        if climb_mode == 3:
            self.find_climb()
        for node in self.nodes:
            if not node.fixed:
                moving_nodes.append(node)

        if climb_mode == 1:
            node = moving_nodes[0]
            save ="%.4f.climb.opt" % node.param
            save = os.path.join(node.dir_pre, save)
            climb_force = node.climb_forces
            forces.append(climb_force)

            opt = get_optimizer(self.control, self.control.climb_optimizer, save)
            opt.initialize()
            opt.load()
            node.positions = opt.step(node.positions,
                                      climb_force)
            opt.dump()
        else:
            #include two fixed end points, if they exist
            if moving_nodes[0].prev:
                moving_nodes.insert(0,moving_nodes[0].prev)
            if moving_nodes[-1].next:
                moving_nodes.append(moving_nodes[-1].next)
            new_t = []
            new_pos = []
            climb_ind = None
            #get all the forces and positions
            #find the index of the climbing image
            for i,node in enumerate(moving_nodes):
                forces.append(node.climb_forces)
                positions.append(node.positions)
                new_t.append(node.param)
                if node.climb:
                    climb_ind = i
            forces = np.array(forces)
            positions = np.array(positions)
            #move nodes, either by global or non-global optimizer
            if self.control.climb_global_opt:
                save = os.path.join(moving_nodes[0].dir_pre, "climbing.opt")
                new_pos = self.g_opt(
                    self.control.climb_optimizer,
                    positions,
                    forces,
                    save)
            else:
                new_pos = self.nong_opt(
                    self.control.climb_optimizer,
                    moving_nodes,
                    positions,
                    forces,
                    ".climb.opt")
            old_t = (get_t(new_pos[0:climb_ind+1]) 
                     * (moving_nodes[climb_ind].param-moving_nodes[0].param) 
                     + moving_nodes[0].param)
            old_t2 = (get_t(new_pos[climb_ind:])
                      * (moving_nodes[-1].param - moving_nodes[climb_ind].param) 
                      + moving_nodes[climb_ind].param)
            old_t = np.append(old_t, old_t2[1:])
            if isinstance(opt, FDOptimize) and opt.finitie_diff:
                new_pos = spline_pos(new_pos, new_t, old_t = old_t)
            for i, position in enumerate(new_pos):
                moving_nodes[i].positions = position
        
        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5
        
        return np.nanmax(forces)

    def g_opt(self, opt_key, positions, forces, save):
        opt = get_optimizer(self.control, opt_key, save)
        opt.initialize()
        opt.load()
        new_pos = opt.step(positions, forces)
        opt.dump()
        return new_pos
    
    def nong_opt(self, opt_key, nodes, positions, forces, save_suffix):
        import os
        new_pos = []
        for i,node in enumerate(nodes):
            save = "%.4f" % node.param
            save += save_suffix
            save = os.path.join(node.dir_pre, save)
            opt = get_optimizer(self.control, opt_key, save)
            opt.initialize()
            opt.load()
            new_pos.append(opt.step(positions[i],
                                    forces[i]))
            opt.dump()
        return new_pos


    def n_nodes(self):
        """
        return the current number of nodes in the path
        """
        return len(self.nodes)

    def interpolate(self, n):
        """
        interpolate/resample the current path
        """
        from aimsChain.interpolate import spline_pos
        import copy
        #if we have only two image, then insert them all
        if self.n_nodes() == 2:
            for i in np.linspace(0,1,n+2)[1:-1]:
                self.insert_node(i)
        #if we want to resample
        #we first add all new coord
        #then remove those that are in orignal but not new
        #list of parameters
        else:
            old_t = []
            new_t = np.linspace(0,1,n+2)[1:-1]
            for node in self.nodes[1:-1]:
                old_t.append(node.param)
            for i in new_t:
                if not (i in old_t):
                    self.insert_node(i)
            for i in old_t:
                if i in new_t:
                    old_t.remove(i)
            for node in self.nodes[1:-1]:
                if node.param in old_t:
                    self.nodes.remove[node]
            
    
    def write_path(self, file="path.dat"):
        """
        Write the current path into a file
        Will contain:
        current runs
        each node's param, dir, and whether they are fixed or not
        This was preferred over pickling for the sake of easy debugging
        May change to pickling in the production level
        """
        data = open(file, 'w')
        data.write('#======================================#\n')
        data.write('#=               Path Info            =#\n')
        data.write('#======================================#\n')
        data.write('%d\n' % self.runs)
        for node in self.nodes:
            data.write('Node\n')
            data.write('%.8f\n' % node.param)
            data.write(node.dir + '\n')
            data.write('%d\n' % int(node.fixed))
            data.write('%d\n' % int(node.climb))
        
        data.close()
    def read_path(self, file="path.dat"):
        """
        Read the file into current path
        """
        nodes = []
        data = open(file, 'r')
        
        while True:
            line = data.readline().replace('\n','')
            if not line:
                break
            if line.split()[0][0] == '#':
                continue
            else:
                self.runs = int(float(line))
                break
    
        while True:
            line = data.readline().replace('\n','')
            if not line:
                break
            if line.split()[0][0] == '#':
                continue
            elif 'Node' in line:
                tmp_node = Node()
                tmp_node.param = float(data.readline().replace('\n',''))
                tmp_node.dir = data.readline().replace('\n','')
                tmp_node.fixed = bool(int(float(data.readline().replace('\n',''))))
                tmp_node.climb = bool(int(float(data.readline().replace('\n',''))))                

                nodes.append(tmp_node)
        self.nodes = nodes


def get_optimizer(control, key, data_name):
        from aimsChain.optimizer.lbfgs import LBFGS
        from aimsChain.optimizer.newbfgs import BFGS
        from aimsChain.optimizer.dampedbfgs import dampedBFGS
        from aimsChain.optimizer.fire import FIRE
        from aimsChain.optimizer.cg import CG

        opt = None
        if key.lower() == "lbfgs":
            opt = LBFGS(data_name, 
                        maxstep=control.lbfgs_maxstep, 
                        memory = control.lbfgs_memory, 
                        alpha = control.lbfgs_alpha)
        elif key.lower() == "bfgs":
            opt = BFGS(data_name,
                       maxstep = control.bfgs_maxstep,
                       alpha = control.bfgs_alpha)
        elif key.lower() == "fire":
            opt = FIRE(data_name,
                       dt = control.fire_dt,
                       maxstep = control.fire_maxstep,
                       dtmax = control.fire_dtmax,
                       Nmin = control.fire_nmin,
                       finc = control.fire_finc,
                       fdec = control.fire_fdec,
                       astart = control.fire_astart,
                       fa = control.fire_fa,
                       a = control.fire_a)
        elif key.lower() == "cg":
            opt = CG(data_name)
        else:
            opt = dampedBFGS(data_name,
                             maxstep = control.bfgs_maxstep,
                             alpha = control.bfgs_alpha)
        
        return opt
