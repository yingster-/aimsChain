"""
This module defines the basic Atom object.
Atoms is the basic geometry object, simply a list of Atom
"""

import numpy as np


class Atom(object):
    """
    Class for a single atom
    Parameters:
    position: 3 floats for xyz
    force: 3 floats for xyz forces
    extra: list of string to store extra info from geometry files
    constraint: mask for constraint, 0 values will be masked
    """
    
    def __init__(self, positions=(0,0,0),
                 forces=(0,0,0),
                 symbol='',
                 constraint=[1,1,1]):
        self.__positions = np.array(positions)
        self.__forces = np.array(forces)
        self.__extra = []
        self.__symbol = symbol
        self.__constraint=np.array(constraint)

    @property
    def positions(self):
        """get positions, numpy array with 3 float"""
        return self.__positions
    @positions.setter
    def positions(self, positions):
        """set positions"""
        self.__positions = np.array(positions)
        
    @property
    def forces(self):
        """
        get forces, numpy array with 3 float
        map constraint over the forces
        """
        return self.__forces*self.__constraint
    @forces.setter
    def forces(self, forces):
        """set forces"""
        self.__forces = np.array(forces)

    @property
    def extra(self):
        """get extra, list of strings"""
        return self.__extra
    @extra.setter
    def extra(self, extra):
        """set the extra"""
        self.__extra = extra
    def add_extra(self,extra):
        """add a new line to the end of existing extra"""
        self.__extra.append(extra)

    @property
    def symbol(self):
        """set the symbol for the atom"""
        return self.__symbol
    @symbol.setter
    def symbol(self,symbol):
        """get symbol for the atom"""
        self.__symbol=symbol

    @property
    def constraint(self):
        """
        get the constraint of the geometry
        """
        return self.__constraint
    @constraint.setter
    def constraint(self, constraint):
        """
        set the constraint for geometry
        constraint can either be combination of x,y,z, or all
        or a array of 3 item for x,y,z axes, either 1 or 0
        1 is free, 0 is fixed
        """
        if isinstance(constraint, basestring):
            if 'x' in constraint:
                self.__constraint = self.constraint&np.array([0,1,1])
            if 'y' in constraint:
                self.__constraint = self.constraint&np.array([1,0,1])
            if 'z' in constraint:
                self.__constraint = self.constraint&np.array([1,1,0])
            if constraint == 'all':
                self.__constraint = np.array([0,0,0])
        else:
            self.__constraint = np.array(constraint)


class Atoms(object):
    """
    Class for a single geometry
    Parameters:
    
    atoms: list of Atom object, basis of Atoms
    lattice: the lattice vectors, for periodic system
    ener: energy of the geometry
    """
    
    def __init__(self, atoms=[], 
                 lattice=None,
                 ener = 0):
        import copy
        self.__atoms=[]
        for atom in atoms:
            if isinstance(atom, Atom):
                self.__atoms.append(atom)
        self.__lattice = None
        if np.shape(lattice) == (3,3):
           self.__lattice = copy.deepcopy(lattice) 
        self.__ener = float(ener)

    @property
    def atoms(self):
        """
        get a list of all atoms in the geometry
        """
        return self.__atoms
    @atoms.setter
    def atoms(self, atom):
        """
        add a new atom to the end of the current list if atom is a single atom
        or else set atoms to the list of atom
        """
        if isinstance(atom, Atom):
            self.__atoms.append(atom)
        elif isinstance(atom, list):
            self.__atoms = atom
    
    @property
    def forces(self):
        """
        get a list of forces, 
        index correspond to position of atom in the list
        """
        forces = []
        for atom in self.__atoms:
            forces.append(atom.forces)
        return np.array(forces)
    @forces.setter
    def forces(self, forces):
        """
        set forces of all atoms
        no.of forces must match no. of atoms
        """
        for i, triplet in enumerate(forces):
            self.__atoms[i].forces = triplet

    @property
    def positions(self):
        """
        get a list of positions,
        index correspond to position of atom in the list
        """
        positions = []
        for atom in self.__atoms:
            positions.append(atom.positions)
        return np.array(positions)
    @positions.setter
    def positions(self, positions):
        """
        set positions of all atoms
        no. of positions must match no. of atoms
        """
        for i, triplet in enumerate(positions):
            self.__atoms[i].positions = triplet

    @property
    def lattice(self):
        """
        get the lattice constant stored in the geometry
        """
        return self.__lattice
    @lattice.setter
    def lattice(self,lattice):
        """
        set the lattice vector for the geometry
        """
        if len(lattice) == 3 and len(lattice[0]) == 3:
            self.__lattice = np.array(lattice)
    @property
    def constraints(self):
        """
        get a list of constraints,
        index correspond to position of atom in the list
        """
        constraints = []
        for atom in self.__atoms:
            constraints.append(atom.constraint)
        return np.array(constraints)
    @constraints.setter
    def constraints(self, constraints):
        """
        set the constraints for all atom at once
        no. of constraints must match no. of atoms
        """
        for i, single_constraint in enumerate(constraints):
            self.__atoms[i].constraint = single_constraint
    @property
    def ener(self):
        """
        get the energy of this geometry
        default value is 0, if not set otherwise
        """
        return self.__ener
    @ener.setter
    def ener(self, value):
        """
        Set the energy of this geometry
        """
        self.__ener = float(value)
