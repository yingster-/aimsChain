"""
This module defines the interpolation functions
All are based on the interpolater provided by scipy
There are: 
A linear interpolater to generate positions between two points
A cubic spline interpolater to resample a set of positions and provide derivatives
A cubic spline interpolater that also takes forces and provide its derivatives
"""

def linear_interp(pos1, pos2, n):
    """
    interpolate n points between pos1 and pos2
    pos1 and pose2 should be a list of triplets
    will return:
    a list of list of triplets with n+2 items (including end points)
    a list of values between 0 and 1, the parametric parameter
    """
    import numpy as np
    from scipy import interpolate
    
    n_atoms = len(pos1)
    pos = np.array([pos1,pos2])

    #this will transform the shape of coord
    #to a list of triplets, describing motion of one atom over entire path
    pos = np.reshape(np.transpose(np.reshape(pos, (2,-1))),(-1,3,2))
    new_t = np.linspace(0,1,n+2)
    new_pos = []
    for cord in pos:
        tck = interpolate.splprep(u = [0,1], x=cord, k=1, s=0)[0]
        new_pos.append(interpolate.splev(new_t, tck))        
    new_pos = np.array(new_pos)
    new_pos = np.reshape(np.transpose(np.reshape(new_pos,(3*n_atoms,-1))), (n+2, n_atoms, 3))
    return np.array(new_pos), new_t


def spline_pos(positions, new_t, old_t=None, k = 3, derv = 0):
    """
    Resample the list of positions with a list of
    parametric parameter t.
    The list of positions is parameterized 
    derv set the derivative to be taken. if 1, will take 1st derivative
    Will return: 
    A list of interpolated positions, matching the specified t
    """
    import numpy as np
    from scipy import interpolate

    positions = np.array(positions)
    if old_t == None:
        old_t = get_t(positions)
    new_t = np.array(new_t)
    n_atoms = len(positions[0])
    n_nodes = len(positions)

    #this will transform the shape of coord
    #to a list of triplets, describing motion of one atom over entire path
    pos = np.reshape(np.transpose(np.reshape(positions, (n_nodes, 3*n_atoms))),(-1,3,n_nodes))
    new_pos = []
    for cord in pos:
        tck = interpolate.splprep(u=old_t,x=cord, k=k, s=0)[0]
        new_pos.append(interpolate.splev(new_t, tck, der=derv))
    new_pos = np.array(new_pos)
    #transform the array back to list of geometries
    new_pos = np.reshape(np.transpose(np.reshape(new_pos,(3*n_atoms,-1))), (-1, n_atoms, 3))
    result = new_pos
    return result

"""
!See how to do this when writting optimizer
Maybe we need something more complex...
Or perhaps interpolating forces is just too naive

def spline_posfor(positions, forces, t, derv = False):
    #Resample the list of positions with a list of
    #parametric parameter t.
    #The list of positions is parameterized 
    #Will return: (doublet or quadruplet)
    #A list of interpolated positions, matching the specified t
    #A list of position derivatives at those point. (Turned on with no_derv)
    #A list of interpolated forces, matching the specified t
    #A list of forces derivatives at those point. (Turned on with no_derv)
    import numpy as np
    from scipy import interpolate

    positions = np.array(positions)

    old_t = get_t(positions)
    new_t = t
    k = 3
    n_atoms = len(positions[0])
    n_nodes = len(positions)

    if len(positions) == 3:
        k = 2
    pos = np.transpose(np.reshape(positions, (n_nodes, 3*n_atoms)))
    tck = interpolate.splprep(pos, u = old_t, k=k, s=0)[0]
    new_pos = interpolate.splev(new_t, tck, der=0)
    new_pos = np.reshape(np.transpose(new_pos), (n_nodes, n_atoms, 3))
    result = new_pos
    if derv:
        new_pos_derv = interpolate.splev(new_t, tck, der=0)
        new_pos_derv = np.reshape(np.transpose(new_pos), (n_nodes, n_atoms, 3))
        result = (result, new_pos_derv)
    return result
"""


def get_t(positions):
    """
    Parameterize positions by
    1.get total path length
    2.use the partial sum/total length as t
    Will return:
    A list of t, from 0 to 1
    """
    import numpy as np
    positions = np.array(positions)
    t=[]
    current_t = 0.
    total_t = [0]
    for i in range(len(positions))[1:]:
        diff = positions[i] - positions[i-1]
        current_t += np.linalg.norm(diff)
        total_t.append(current_t)
    for i in total_t:
       t.append(i/total_t[-1])
    return np.array(t)
        
    



def get_total_length(positions):
    """
    Return the total length of positions
    Calculated using numpy norm
    """
    import numpy as np
    length = 0
    positions = np.array(positions)
    for i in range(len(positions))[1:]:
        diff = positions[i] - positions[i-1]
        length += np.linalg.norm(diff)
    return length
        
    
