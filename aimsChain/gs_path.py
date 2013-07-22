"""
The growing string path
"""
import numpy as np
from aimsChain.utility import vmag, vunit, vproj
from aimsChain.path import Path, get_optimizer
from aimsChain.node import Node
from aimsChain.optimizer.optimize import FDOptimize
class GrowingStringPath(Path):

    def __init__(self, 
             nodes = [],
             control = None):

        super(GrowingStringPath,self).__init__(nodes,control)

        self.lower_end = 0

    @property
    def upper_end(self):
        return self.lower_end + 1
    @property
    def linsep(self):
#        print "using " + str(self.control.nimage) + " images\n"
        return 1.0/(self.control.nimage + 1)

    @property
    def params(self):
        t = []
        for node in self.nodes:
            t.append(node.param)
        return t
    @property
    def lower(self):
        return self.nodes[self.lower_end]
    @property
    def upper(self):
        return self.nodes[self.upper_end]


    def add_lower(self):
        from aimsChain.interpolate import spline_pos
        import copy
        if ((self.upper.param - self.lower.param)
            <= self.linsep * 1.5):
            return False
        if (self.n_nodes() < 4) or True:
            self.insert_node(self.lower.param + self.linsep)
            self.lower_end += 1
            new_t = []
            positions = []
            for node in self.nodes:
                new_t.append(node.param)
                positions.append(node.positions)
            positions = spline_pos(np.array(positions), new_t)
            for i,pos in enumerate(positions):
#                if i <= self.lower_end:
                self.nodes[i].positions = pos
            return True
        diff = self.nodes[self.lower_end].positions - self.nodes[self.lower_end-1].positions

        new_node = Node(param = self.lower.param + self.linsep,
                        geometry = copy.deepcopy(self.nodes[0].geometry),
                        path=self)
        new_node.positions = self.lower.positions + diff
        new_node.update_dir()

        self.nodes = new_node

        self.lower_end += 1
        return True
        
    def add_upper(self):    
        import copy
        from aimsChain.interpolate import spline_pos
        if ((self.upper.param - self.lower.param)
            <= self.linsep * 1.5):
            return False
        if (self.n_nodes() < 4) or True:
            self.insert_node(self.upper.param - self.linsep)
            
            new_t = []
            positions = []
            for node in self.nodes:
                new_t.append(node.param)
                positions.append(node.positions)
            positions = spline_pos(np.array(positions), new_t)
            for i,pos in enumerate(positions):
#                if i >= self.upper_end:
                self.nodes[i].positions = pos
            return True

        diff = self.nodes[self.upper_end+1].positions - self.nodes[self.upper_end].positions
        new_node = Node(param = self.upper.param-self.linsep,
                        geometry = copy.deepcopy(self.nodes[0].geometry),
                        path=self)
        new_node.positions = self.upper.positions - diff
        new_node.update_dir()

        self.nodes = new_node
        return True

    def lower_tangent(self):
        from aimsChain.interpolate import spline_pos
        if (self.upper.param - self.lower.param) <= self.linsep*1.5:
            return self.lower.get_tangent()
        positions = []
        param = []
        loc = self.lower_end
        for i in self.nodes:
            positions.append(i.positions)
            param.append(i.param)
          
        derv = spline_pos(positions, param, param, 3, 1)[loc]
        tangent = derv
        
        return vunit(tangent)

    def upper_tangent(self):
        if (self.upper.param - self.lower.param) <= self.linsep*1.5:
            return self.upper.get_tangent()
        from aimsChain.interpolate import spline_pos
        positions = []
        param = []
        loc = self.upper_end
        for i in self.nodes:
            positions.append(i.positions)
            param.append(i.param)
          
        derv = spline_pos(positions, param, param, 3, 1)[loc]
        tangent = derv
        
        return vunit(tangent)

    def lower_force(self):
        forces = self.lower.forces
        tangent = self.lower_tangent()
        forces -= vproj(forces,tangent)
        return forces

    def upper_force(self):
        forces = self.upper.forces
        tangent = self.upper_tangent()
        forces -= vproj(forces,tangent)
        return forces

    def lower_residual(self):
        forces = self.lower_force()
        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5

        return np.nanmax(forces)

    def upper_residual(self):

        forces = self.upper_force()
        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5

        return np.nanmax(forces)



    def move_nodes(self):
        forces = []
        if (self.upper.param - self.lower.param) <= self.linsep*1.5:
            forces.append(self.move_nodes_combined())
        else:
            forces.append(self.move_nodes_separated())

        return max(forces)

    def move_nodes_separated(self):
        from aimsChain.interpolate import spline_pos, get_total_length, get_t
        import os
        
        positions = []
        forces = []
        new_pos = []
        new_pos2= []
        total_length = 0
        for node in self.nodes:
            positions.append(node.positions)
            if node is self.upper:
                forces.append(self.upper_force())
            elif node is self.lower:
                forces.append(self.lower_force())
            else:
                forces.append(node.normal_forces)

        forces = np.array(forces)
        positions = np.array(positions)
        

        new_pos,opt = self.nong_opt(
            self.control.optimizer,
            self.nodes,
            positions,
            forces,
            ".gs.opt")

        lower_length = get_total_length(new_pos[:self.lower_end+1])
        upper_length = get_total_length(new_pos[self.upper_end:])
        total_length = get_total_length(new_pos)

        temp_pos = new_pos[:self.lower_end+1]
        old_t = get_t(temp_pos)
        new_t = np.linspace(0.0,
                            (total_length*self.lower.param)/lower_length,
                            len(temp_pos))
        

        
        if not (isinstance(opt, FDOptimize) and opt.finite_diff):
            temp_pos = spline_pos(temp_pos, new_t, old_t)
        new_pos2.extend(temp_pos)

        temp_pos = new_pos[self.upper_end:]
        old_t = get_t(temp_pos)-1
        new_t = np.linspace(-1*(total_length*(1-self.upper.param))/upper_length,
                            0,
                            len(temp_pos))

#        log = open("logfile",'a')
        
#        log.write(str(isinstance(opt,FDOptimize))+'\n')
#        if (not isinstance(opt, FDOptimize)) or opt.finite_diff:
#            log.write("reparamet\n")
        if (not isinstance(opt, FDOptimize)) or opt.finite_diff:
            temp_pos = spline_pos(temp_pos, new_t, old_t)
#        else:
#            log.write("noreparamet\n")

 #       log.close()
        new_pos2.extend(temp_pos)


        for i,position in enumerate(new_pos):
            self.nodes[i].positions = position


        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5
        return np.nanmax(forces)




    def move_nodes_combined(self):
        from aimsChain.interpolate import spline_pos, get_total_length
        import os
        
        positions = []
        forces = []
        new_t = self.params
        new_pos = []

        for node in self.nodes:
            positions.append(node.positions)
            forces.append(node.normal_forces)

        forces = np.array(forces)
        positions = np.array(positions)
        
        new_pos,opt = self.nong_opt(
            self.control.optimizer,
            self.nodes,
            positions,
            forces,
            ".gs.opt")
#        log = open("logfile",'w')
        if not (isinstance(opt, FDOptimize) and opt.finite_diff):
#            log.wirte("reparamet\n")
            new_pos = spline_pos(new_pos, new_t)
#        else:
#            log.write("no reparamet\n")
#        log.close()

        for i,position in enumerate(new_pos):
            self.nodes[i].positions = position


        forces = np.reshape(forces, (-1,3))
        forces = np.sum(forces**2,1)**0.5
        return np.nanmax(forces)
