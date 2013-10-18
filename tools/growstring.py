#!/usr/bin/env python

import subprocess
import os
import distutils.dir_util as dir_util
import shutil
import numpy as np
import cPickle as cp

from aimsChain.gs_path import GrowingStringPath
from aimsChain.node import Node
from aimsChain.aimsio import read_aims
from aimsChain.config import Control
from aimsChain.interpolate import get_t
from aimsChain.aimsio import write_mapped_aims, write_xyz, write_aims

def run_aims(paths):
    global control
    while len(paths) > 0:
        path = paths[0]
        if control.restart:
            save_restart(paths)
        #remove the ending slash if it exist
        if path[-1] == "/":
            path = path[:-1]
        #generate the name for output
        filename = path[len(path)-path[::-1].index('/'):]+'.out'
        
        command = 'cd ' + path + ';' + control.run_aims + ' > ' + filename
        subprocess.call(command, shell=True)
        paths.remove(path)
        if control.restart:
            save_restart(paths)
        
    paths = []


def initial_interpolation():
    global control
    global path
    #set the initial and final image
    ininode = Node(param = 0.0, 
                   geometry = read_aims(control.ini))
    finnode = Node(param = 1.0,
                   geometry = read_aims(control.fin))

    #does periodic transformation between initial and final node
    #minimize distance between initial and final image atom-wise
    #maybe this should be moved to another file...runchain is getting too messy
    #leave it like this for now
    if control.periodic_interp and finnode.geometry.lattice != None:
        lattice = finnode.geometry.lattice
        initial_pos = ininode.positions
        curr_pos = finnode.positions
        fin_pos = []
        for i,atom_pos in enumerate(curr_pos):
            dis = 1000000
            pos = None
            for a in [-1,0,1]:
                for b in [-1,0,1]:
                    for c in [-1,0,1]:
                        pos_tmp = (atom_pos +
                                   a*lattice[0] + b*lattice[1] + c*lattice[2])
                        dis_tmp = np.sum(np.array(pos_tmp - initial_pos[i])**2)**0.5
                        if dis_tmp <= dis:
                            pos = pos_tmp
                            dis = dis_tmp
            fin_pos.append(pos)

        finnode.positions = fin_pos  

    ininode.fixed = True
    finnode.fixed = True

    path.nodes = [ininode, finnode]

    path.add_lower()
    path.add_upper()


def save_restart(path):
    global restart_stage
    global force
    restart = open("iterations/restart_file", 'w')
    cp.dump((path, restart_stage, force), restart)
    restart.close()

def read_restart():
    global path_to_run
    global force
    global restart_stage
    global path
    file_exist = os.path.isfile("iterations/restart_file")
    if file_exist:
        restart = open("iterations/restart_file", 'r')
        path_to_run, restart_stage, force = cp.load(restart)
        path.read_path("iterations/path.dat")
        path.load_nodes()
        restart.close()
    return file_exist

def write_current():
    global path
    dir_name = "iteration%04d" % path.runs
    dir_name = os.path.join("paths", dir_name)
    try:
        os.mkdir(dir_name)
    except OSError:
        pass
    ener = []
    climb = []
    fixed = []
    write_xyz(os.path.join(dir_name, "path.xyz"), path, control.xyz_lattice)
    for i,node in enumerate(path.nodes):
        ener.append(node.ener)
        climb.append(node.climb)
        fixed.append(node.fixed)
        i += 1
        file_name = os.path.join(dir_name, "image%03d.in" % i)
        if control.map_unit_cell:
            write_mapped_aims(file_name, node.geometry)
        else:
            write_aims(file_name, node.geometry)
    ener = np.array(ener) - ener[0]
    enerfile = open(os.path.join(dir_name, "ener.lst"), 'w')
    for i,energy in enumerate(ener):
        enerfile.write("image%03d\t%.10f" % (i+1, energy))
        if climb[i]:
            enerfile.write("\t CLIMB")
        if fixed[i]:
            enerfile.write("\t FIXED")
        enerfile.write("\n")
    enerfile.close()
    pathfile = open(os.path.join(dir_name, "path.lst") ,'w')
    for i,item in enumerate(path.get_paths()):
        pathfile.write("image%03d\t%s" % (i+1,item))
        if climb[i]:
            pathfile.write("\t CLIMB")
        if fixed[i]:
            pathfile.write("\t FIXED")
        pathfile.write("\n")
    pathfile.close()
        
    

force = 10.0
growing = True
control = Control()
path = GrowingStringPath(control=control)
restart_stage = "growing"
is_restart = control.restart and read_restart()

forcelog = open("growing_forces.log", 'a')
if restart_stage != "growing":
    growing = False


if not is_restart:
    for directory in ["paths", "iterations", "grownstring"]:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
    os.mkdir("paths")

    
    initial_interpolation()

    #write directory for images
    path_to_run = path.write_all_node()
    path.write_path("iterations/path.dat")

    forcelog.write("Iteration\tResidual force\t\tLower end force\t\tUpper end force \n")
    forcelog.flush()

while growing:

    run_aims(path_to_run)
    path.load_nodes()
    force,low_force,high_force = path.move_nodes()
    write_current()
    curr_runs = path.runs

    if low_force < control.gs_thres:
        growing = path.add_lower() and growing
    if high_force < control.gs_thres:
        growing = path.add_upper() and growing

    path.add_runs()
    path_to_run = path.write_node()

    forcelog.write('iteration%04d\t%16.8f\t%16.8f\t%16.8f \n' % 
                   (curr_runs,force,low_force, high_force))
    forcelog.flush()
    path.write_path("iterations/path.dat")
    

run_aims(path_to_run)
write_current()
path.write_path("iterations/path.dat")


force = 10.0
forcelog.write("Path is grown.\n")

forcelog.close()

restart_stage = "grown"
if control.restart:
    save_restart([])

try:
    os.mkdir('grownstring')
except OSError:
    pass

for i,dir in enumerate(path.get_paths()):
    i+=1
    dir_util.copy_tree(dir, os.path.join('grownstring', "image%03d" % i))
write_xyz("grownstring/path.xyz", path, control.xyz_lattice)
