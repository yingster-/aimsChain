#!/usr/bin/env python

import subprocess
import os
import distutils.dir_util as dir_util
import shutil
import numpy as np
import cPickle as cp

from aimsChain.string_path import StringPath
from aimsChain.neb_path import NebPath
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


    #fix the initial and final node    
    ininode.fixed = True
    finnode.fixed = True

    #parse the externl geometry
    try:
        nodes = [ininode]
        if control.ext_geo and os.path.isfile(str(control.ext_geo)):
            geo = open(control.ext_geo)
            lines = geo.readlines()
            geo.close()
            for line in lines:
                inp = line.split()
                if inp == []:
                    continue
                elif inp[0][0] == '#':
                    continue
                else:
                    if os.path.isfile(inp[0]):
                        nodes.append(Node(param = 0.5, 
                                          geometry = read_aims(inp[0]), 
                                          fixed = False))
            nodes.append(finnode)
            geos = []
            for node in nodes:
                geos.append(node.positions)
            t = get_t(geos)
            for i in range(len(t)):
                nodes[i].param = t[i]
            path.nodes = nodes
            #if resample is turned on and n_image does not equal to 
            #number of images from external source,
            #then resample the path
            if control.resample and control.nimage != (len(nodes)-2):
                path.interpolate(control.nimage)

    except:
        print '!Error interprating the external geometries\n'
        print '!Using standard interpolation method for initial geometries\n'
        nodes = [ininode, finnode]

    #if there were no external geometry, linear interpolate the image
    if len(nodes) <= 2:
        path.nodes = [ininode, finnode]
        #interpolate the images
        path.interpolate(control.nimage)

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
control = Control()
if control.method == "neb":
    path = NebPath(control=control)
else:
    path = StringPath(control=control)
restart_stage = 0
is_restart = read_restart() and control.restart


forcelog = open("forces.log", 'a')


if not is_restart:
    for directory in ["paths", "iterations", "optimized"]:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
    os.mkdir("paths")

    
    initial_interpolation()

    #write directory for images
    path_to_run = path.write_all_node()
    path.write_path("iterations/path.dat")

    forcelog.write("#Residual Forces in the system:\n")
    forcelog.flush()

if restart_stage == 0:
    while force > control.thres:
        run_aims(path_to_run)
        path.load_nodes()
        force = path.move_nodes()
        write_current()
        curr_runs = path.runs
        if force > control.thres:
            path.add_runs()
            path_to_run = path.write_node()
        forcelog.write('iteration%04d\t%16.16f \n' % (curr_runs,force))
        forcelog.flush()
        path.write_path("iterations/path.dat")
    force = 10.0
    forcelog.write("System has converged.\n")

forcelog.close()

if control.use_climb:
    forcelog = open("climbing_forces.log", 'a')
    if restart_stage != 1:
        path.find_climb()
        path.add_runs()
        if control.climb_control != "control.in":
            path_to_run = path.write_all_node(control.climb_control)
        else:
            path_to_run = path.write_node()

        forcelog.write("#Residual Forces in the Climbing image:\n")
        forcelog.flush()
        restart_stage = 1


    while force > control.climb_thres:
        run_aims(path_to_run)
        path.load_nodes()
        force = path.move_climb()
        write_current()
        curr_runs = path.runs
        if force > control.climb_thres:
            path.add_runs()
            path_to_run = path.write_node(control.climb_control)
        forcelog.write('iteration%04d\t%16.16f \n' % (curr_runs, force))
        forcelog.flush()
        path.write_path("iterations/path.dat")
    forcelog.write('Climbing image has converged.\n')
    forcelog.close()

try:
    os.mkdir('optimized')
except OSError:
    pass

for i,dir in enumerate(path.get_paths()):
    i+=1
    dir_util.copy_tree(dir, os.path.join('optimized', "image%03d" % i))
write_xyz("optimized/path.xyz", path, control.xyz_lattice)
