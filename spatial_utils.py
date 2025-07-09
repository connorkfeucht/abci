import math

# return true if the objects overlap 
def overlap(b1, b2, min_sep=0.0): # b = (xmin, xmax, ymin, ymax, zmin, zmax)
    x_overlap = (b1[0] < b2[1] + min_sep) and (b1[1] + min_sep > b2[0])
    y_overlap = (b1[2] < b2[3] + min_sep) and (b1[3] + min_sep > b2[2])
    z_overlap = (b1[4] < b2[5] + min_sep) and (b1[5] + min_sep > b2[4])

    return x_overlap and y_overlap and z_overlap

# computes the distance between two objects, assumes they do not overlap
def euclidean_distance(b1, b2):
    if b1[0] > b2[1]: # b1's xmin greater than b2's xmax
        dx = b1[0] - b2[1] # dist = b1's xmin - b2's xmax
    elif b2[0] > b1[1]: # b2's xmin greater than b1's xmax
        dx = b2[0] - b1[1] # dist = b2's xmin - b1's xmax
    else: # overlap
        dx = 0

    if b1[2] > b2[3]:
        dy = b1[2] - b2[3]
    elif b2[2] > b1[3]:
        dy = b2[2] - b1[3]
    else:
        dy = 0

    if b1[4] > b2[5]:
        dz = b1[4] - b2[5]
    elif b2[4] > b1[5]:
        dz = b2[4] - b1[5]
    else:
        dz = 0
    
    euclidean_distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return euclidean_distance