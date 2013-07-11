#!/usr/bin/env python

import subprocess
import os
import distutils.dir_util as dir_util
import shutil
import numpy as np
import cPickle as cp

from aimsChain.node import Path
from aimsChain.node import Node
from aimsChain.aimsio import read_aims
from aimsChain.config import Control
from aimsChain.interpolate import get_t
from aimsChain.aimsio import write_mapped_aims, write_xyz, write_aims


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

control = Control()
path = Path(control=control)

if os.path.isdir("interpolation"):
    shutil.rmtree("interpolation")
os.mkdir("interpolation")

initial_interpolation()

write_xyz(os.path.join("interpolation", "path.xyz"),path, control.xyz_lattice)

for i,node in enumerate(path.nodes):
    i += 1
    file_name = os.path.join("interpolation", "image%03d.in" % i)
    write_aims(file_name, node.geometry)

