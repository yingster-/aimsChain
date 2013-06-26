"""
This module defines the Node and the Path
Node is intended to be the fundamental unit in string method
Unlike geometry, it maintains parametric values, which remains 
unchanged through the entire calculation
Path is a collection of nodes, with some additional functions
"""

import numpy as np
from aimsChain.utility import vmag, vunit, vproj
from aimsChain.config import Control
class Node(object):
    """
    Class for a single Node
    Parameters:
    geometry: the current Atoms object for the node
    path: the path that the particular node blongs to
    dir: the current dir that the node reads
    """
    def __init__(self, 
                 param = 0,
                 geometry = None,
                 path = None,
                 dir = None,
                 fixed = False):
        from aimsChain.atom import Atoms
        self.__param = float(param)
        self.__geometry = geometry
        if self.__geometry == None:
            self.__geometry = Atoms()
        self.__path = path
        self.__dir = dir
        self.__fixed = fixed
        self.__climb = False
        self.__dir_pre = "iterations"
        self.__previous_dir = None

    @property
    def param(self):
        return self.__param
    @param.setter
    def param(self, t):
        self.__param = float(t)
    @property
    def geometry(self):
        return self.__geometry
    @geometry.setter
    def geometry(self, atoms):
        self.__geometry = atoms
    @property
    def climb(self):
        return self.__climb
    @climb.setter
    def climb(self, climb):
        self.__climb = bool(climb)
    
    @property
    def control(self):
        if self.path != None:
            return self.path.control
        else:
            return config()

    @property
    def path(self):
        return self.__path
    @path.setter
    def path(self, path):
        self.__path = path

    @property
    def dir(self):
        return self.__dir
    @dir.setter
    def dir(self, dir):
        self.__dir = dir
    
    @property
    def dir_pre(self):
        return self.__dir_pre
    @dir_pre.setter
    def dir_pre(self,dir_pre):
        self.__dir_pre = dir_pre
    @property
    def ener(self):
        return float(self.__geometry.ener)
    @ener.setter
    def ener(self, ener):
        self.__geometry.ener = float(ener)
    
    @property
    def positions(self):
        return self.__geometry.positions
    @positions.setter
    def positions(self, positions):
        if not self.__fixed:
            self.__geometry.positions = positions
    
    @property
    def forces(self):
        if self.fixed:
            return np.zeros(np.shape(self.geometry.forces))
        else:
            return self.geometry.forces
    @forces.setter
    def forces(self, forces):
        self.__geometry.forces = forces

    @property
    def climb_forces(self):
        if self.fixed:
            return np.zeros(np.shape(self.forces))
        elif self.climb:
            forces = self.forces
            tangent = self.get_tangent()
            return forces - 2*vproj(forces,tangent)
        else:
            if self.control.method == "neb":
                return self.spring_forces
            else:
                return self.normal_forces

    @property
    def normal_forces(self):
        if self.fixed:
            return np.zeros(np.shape(self.geometry.forces))
        forces = self.forces
        tangent = self.get_tangent()
        forces -= vproj(forces, tangent)
        return forces

    @property
    def spring_forces(self):
        if self.path == None:
            k = 10.0
        else:
            k = self.control.spring_k
        if self.fixed:
            return np.zeros(np.shape(self.positions))
        forces = self.normal_forces
        tangent = self.get_tangent()
        tan1 = self.next.positions - self.positions
        tan2 = self.positions - self.prev.positions
        mag = (vmag(tan1) - vmag(tan2))
        forces += k * mag * tangent
        return forces
    
   
    @property
    def ener(self):
        return self.__geometry.ener
    @ener.setter
    def ener(self, energy):
        self.__geometry.ener = energy

    @property
    def fixed(self):
        return self.__fixed
    @fixed.setter
    def fixed(self, value):
        self.__fixed = bool(value)

    @property
    def prev(self):
        """get the previous element in path"""
        if not self.path:
            return None
        i = self.path.nodes.index(self)
        if i == 0:
            return None
        else:
            return self.path.nodes[i-1]
    @property
    def next(self):
        """get the next element in path"""
        if not self.path:
            return None
        i = self.path.nodes.index(self)
        if i+1 == len(self.path.nodes):
            return None
        else:
            return self.path.nodes[i+1]

    @property
    def previous_dir(self):
        """store the previous dir for this node"""
        return self.__previous_dir
    @previous_dir.setter
    def previous_dir(self, value):
        self.__previous_dir = value

    def get_tangent(self, for_climb=False, unit=True):
        prev = self.prev
        next = self.next
        tangent = None
        if prev == None:
            tangent = (next.positions - self.positions)
        elif next == None:
            tangent = (self.positions - prev.positions)
        else:
            tan1 = next.positions - self.positions
            tan2 = self.positions - prev.positions

            if next.ener >= self.ener and self.ener >= prev.ener and not for_climb:
                tangent = tan1
            elif next.ener < self.ener and self.ener < prev.ener and not for_climb:
                tangent = tan2
            else:
                max = np.nanmax((abs(next.ener-self.ener), abs(self.ener-prev.ener)))
                min = np.nanmin((abs(next.ener-self.ener), abs(self.ener-prev.ener)))

                if next.ener >= prev.ener:
                    tangent = (max * tan1 + min * tan2)
                else:
                    tangent = (min * tan1 + max * tan2)
        if unit:
            return vunit(tangent)
        

        return tangent

    def write_node(self, write_fixed = False):
        """
        create a directory for each node
        including geometry and control

        TODO: also get restarts from previous run
        """
        import glob, os, shutil
        from aimsChain.aimsio import write_aims
        dir = os.path.join(self.dir_pre, self.dir)
        if (not self.fixed) or write_fixed:
            if not os.path.isdir(dir):
                os.makedirs(dir)
            write_aims(os.path.join(dir, "geometry.in"), self.geometry)
            shutil.copy("control.in", dir)
            if (self.path and self.control.aims_restart
                and self.previous_dir):
                if self.previous_dir != self.dir:
                    restart_file = self.control.aims_restart + "*"
                    files = glob.glob(os.path.join(self.dir_pre, self.previous_dir, restart_file))
                    for file in files:
                        if os.path.isfile(file):
                            shutil.copy(file, dir)
            return dir
        else:
            return None
    
    def update_dir(self, write_fixed = False):
        """
        update the dirs for all the nodes
        if write_fixed = True, 
        even fixed nodes will get updates
        """
        if (not self.fixed) or write_fixed:
            self.previous_dir = self.dir
            self.dir = "aims-chain-node-%.5f-%06d" % (self.param, self.path.runs)
                
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
    @property
    def nodes(self):
        """get a list of all nodes in the path"""
        return self.__nodes
    @nodes.setter
    def nodes(self,node):
        """take either a single node or a list of nodes
        in case of a single node, insert it into self.nodes according to param
        in case of a list of nodes, set it as self.nodes"""
        if isinstance(node, Node):
            node.path = self
            if len(self.__nodes) == 0 or self.__nodes[-1].param <= node.param:
                self.__nodes.append(node)
            else:
                for i,n in enumerate(self.__nodes):
                    if node.param <= n.param:
                        self.__nodes.insert(i, node)
                        break
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

    def write_node(self):
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
            dir=node.write_node()
            if dir != None:
                path.append(dir)
        return path
    
    def write_all_node(self):
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
            path.append(node.write_node(True))
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

    def move_nodes(self, t_step = 0.03):
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
            self.__nodes[i].positions = position

        if diff_pos != None:
            diff_pos = np.nanmax(diff_pos)
            return diff_pos/t_step
        else:
            return 1.0

    def move_nodes2(self):
        """
        move the nodes according to NEB rules
        """
        import os
        from aimsChain.optimizer.newfire import newFIRE
        from aimsChain.optimizer.euler import EULER
        from aimsChain.interpolate import spline_pos
        positions = []
        forces = []
        new_t = []
        diff_pos = None
        new_pos = []
        t_step = []
        for node in self.nodes:
            new_t.append(node.param)
            if (np.sum(node.positions) != 0):
                positions.append(node.positions)
                forces.append(node.forces)
      
        forces = np.array(forces)
        positions = np.array(positions)
        """
        hess = os.path.join(self.nodes[0].dir_pre, "string.opt")
        opt = newFIRE(hess)
        opt.initialize()
        opt.load()
        new_pos = opt.step(positions, forces)
        opt.dump()
        """
        for i,node in enumerate(self.nodes):
            hess = "%.4f.opt" % node.param
            hess = os.path.join(node.dir_pre, hess)
            opt = newFIRE(hess)
            opt.initialize()
            opt.load()
            next_pos = opt.step(positions[i], forces[i])
            new_pos.append(next_pos)
        #    t_step.append(next_t)
            opt.dump()
        
#        t_step = np.array(t_step)
        new_pos = spline_pos(new_pos, new_t)
        
        if positions.shape == new_pos.shape:
            diff_pos = new_pos - positions
            diff_pos = np.reshape(diff_pos, (-1,3))
            diff_pos = np.sum(diff_pos**2,1)**0.5


        for i,pos in enumerate(new_pos):
            self.nodes[i].positions = pos
            
        if diff_pos != None:
            np.set_printoptions(suppress = True)
            return 1.0
            return np.nanmax(diff_pos)
        else:
            return 1.0



    def move_neb(self):
        """
        move the nodes according to NEB rules
        """
        import os
        from aimsChain.optimizer.lbfgs import LBFGS
        from aimsChain.optimizer.newbfgs import BFGS
        from aimsChain.optimizer.euler import EULER
        from aimsChain.optimizer.fire import FIRE
        positions = []
        forces = []
        new_pos = []
        for node in self.nodes:
            positions.append(node.positions)
            forces.append(node.spring_forces)
      
        forces = np.array(forces)
        positions = np.array(positions)
        if self.control.global_opt:
            print "Using global optimizer."
            hess = os.path.join(self.nodes[0].dir_pre, "neb.opt")
            opt = BFGS(hess, alpha = 70)
            opt.initialize()
            opt.load()
            new_pos = opt.step(positions, forces)
            opt.dump()
        else:
            print "Using non-global optimizer."
            for i,node in enumerate(self.nodes):
                hess = "%.4f.opt" % node.param
                hess = os.path.join(node.dir_pre, hess)
                opt = FIRE(hess)
                opt.initialize()
                opt.load()
                new_pos.append(opt.step(positions[i],
                                        forces[i]))
                opt.dump()

        for i,pos in enumerate(new_pos):
            self.nodes[i].positions = pos
            
        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5

        return np.nanmax(forces)
            
    def move_string(self):
        """
        move the node according to normal force
        the original string method
        """
        from aimsChain.optimizer.lbfgs import LBFGS
        from aimsChain.optimizer.newbfgs import BFGS
        from aimsChain.optimizer.dampedbfgs import dampedBFGS
        from aimsChain.optimizer.fire import FIRE
        from aimsChain.interpolate import spline_pos
        import os
        positions = []
        forces = []
        new_t = []
        new_pos = []
        for node in self.nodes:
            #list all the new params
            new_t.append(node.param)
            #add pos and force to the list
            #only if it is not new (0 been default value)
#            if (np.sum(node.positions) != 0):
            positions.append(node.positions)
            forces.append(node.normal_forces)



        forces = np.array(forces)
        positions = np.array(positions)
        if self.control.global_opt:
            hess = os.path.join(self.nodes[0].dir_pre, "string.opt")
            opt = dampedBFGS(hess)
            opt.initialize()
            opt.load()
            new_pos = opt.step(positions, forces)
            opt.dump()
        else:
            for i,node in enumerate(self.nodes):
                hess = "%.4f.opt" % node.param
                hess = os.path.join(node.dir_pre, hess)
                opt = FIRE(hess)
                opt.initialize()
                opt.load()
                new_pos.append(opt.step(positions[i],
                                        forces[i]))
                opt.dump()

        new_pos = spline_pos(new_pos, new_t)
        for i,position in enumerate(new_pos):
            self.__nodes[i].positions = position

        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5
        return np.nanmax(forces)
            

    def find_climb(self):
        """
        turn on the climb flags in the list of nodes
        """
        import copy
        from aimsChain.interpolate import arb_interp, spline_pos
        energy = []
        climb_nodes = []
        positions = []
        old_t = []
        target_node = None

        for node in self.nodes:
            energy.append(node.ener)
            old_t.append(node.param)
            positions.append(node.positions)
            node.fixed = True

        if self.control.climb_interp == False:
            ind = energy.index(np.nanmax(energy[1:-1]))
            self.nodes[ind].fixed = False
            self.nodes[ind].climb = True
            target_node = self.nodes[ind]
        else:
            new_t = np.linspace(0,1,1001)
            energy_interp = arb_interp(energy, new_t, old_t)
            ind = np.where(energy_interp == np.nanmax(energy_interp[1:-1]))[0][0]
            change_t = np.absolute(old_t - new_t[ind])
            mint_ind = np.where(change_t == np.nanmin(change_t))[0][0]
            if(change_t[mint_ind] <= 0):
                t_thres = old_t[mint_ind+1] - old_t[mint_ind]
            else:
                t_thres = old_t[mint_ind] - old_t[mint_ind-1]
            t_thres = t_thres/4.0
            if np.nanmin(change_t) < t_thres:
                self.nodes[mint_ind].fixed = False
                self.nodes[mint_ind].climb = True
                target_node = self.nodes[mint_ind]
            else:
                new_t = np.array(new_t[ind])
                new_node = Node(param = new_t,
                                geometry = copy.deepcopy(self.nodes[mint_ind].geometry),
                                path = self)
                new_pos = spline_pos(positions, new_t, old_t = old_t)
                new_node.positions = new_pos[0]
                new_node.update_dir()
                new_node.climb = True
                new_node.fixed = False
                new_node.ener = np.nanmax(energy_interp[1:-1])
                target_node = new_node
                self.nodes = new_node
        if self.control.climb_mode == 2:
            target_node.prev.fixed = False
            target_node.next.fixed = False
            


    def move_climb(self):
        """
        move the climbing nodes using BFGS
        """
        from aimsChain.optimizer.fire import FIRE
        from aimsChain.optimizer.newbfgs import BFGS
        from aimsChain.optimizer.dampedbfgs import dampedBFGS
        from aimsChain.interpolate import spline_pos, get_t
        import os
        moving_nodes = []
        forces = []
        positions = []
        for node in self.nodes:
            if not node.fixed:
                moving_nodes.append(node)

        if self.control.climb_mode == 1:
            node = moving_nodes[0]
            hess="%.4f.climb.opt" % node.param
            hess = os.path.join(node.dir_pre, hess)
            climb_force = node.climb_forces
            forces.append(climb_force)

            opt = dampedBFGS(hess)
            opt.initialize()
            opt.load()
            node.positions = opt.step(node.positions,
                                      climb_force)
            opt.dump()
        else:
            if moving_nodes[0].prev:
                moving_nodes.insert(0,moving_nodes[0].prev)
            if moving_nodes[-1].next:
                moving_nodes.append(moving_nodes[-1].next)
            new_t = []
            new_pos = []
            climb_ind = None
            for i,node in enumerate(moving_nodes):
                forces.append(node.climb_forces)
                positions.append(node.positions)
                new_t.append(node.param)
                if node.climb:
                    climb_ind = i
            forces = np.array(forces)
            positions = np.array(positions)
            if self.control.global_opt:
                hess = os.path.join(moving_nodes[0].dir_pre, "climbing.opt")
                opt = dampedBFGS(hess)
                opt.initialize()
                opt.load()
                new_pos = opt.step(positions, forces)
                opt.dump()
            else:
                for i,node in enumerate(moving_nodes):
                    hess = "%.4f.climb.opt" % node.param
                    hess = os.path.join(node.dir_pre, hess)
                    opt = FIRE(hess)
                    opt.initialize()
                    opt.load()
                    new_pos.append(opt.step(positions[i],
                                            forces[i]))
                    opt.dump()
            
            old_t = (get_t(new_pos[0:climb_ind+1]) 
                     * (moving_nodes[climb_ind].param-moving_nodes[0].param) 
                     + moving_nodes[0].param)
            old_t2 = (get_t(new_pos[climb_ind:])
                      * (moving_nodes[-1].param - moving_nodes[climb_ind].param) 
                      + moving_nodes[climb_ind].param)
            old_t = np.append(old_t, old_t2[1:])
            new_pos = spline_pos(new_pos, new_t, old_t = old_t)
            for i, position in enumerate(new_pos):
                moving_nodes[i].positions = position
        
        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5
        
        return np.nanmax(forces)
    def n_nodes(self):
        """
        return the current number of nodes in the path
        """
        return len(self.__nodes)

    def interpolate(self, n):
        """
        interpolate the current path
        will only work if there are currently two nodes in the path
        """
        from aimsChain.interpolate import linear_interp
        import copy
        if self.n_nodes() != 2:
            return False
        pos1 = self.__nodes[0].positions
        pos2 = self.__nodes[1].positions
        positions, new_t = linear_interp(pos1,pos2,n)
        positions = positions[1:-1]
        new_t = new_t[1:-1]
        for i,t in enumerate(new_t):
            new_node = Node(param = t, 
                            geometry = copy.deepcopy(self.__nodes[0].geometry), 
                            path = self)
            new_node.positions = positions[i]
            new_node.update_dir()
            self.nodes = new_node
    
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
