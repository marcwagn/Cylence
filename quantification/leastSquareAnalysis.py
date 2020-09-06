import numpy as np

def fitLine(points):
    """ fits line using L2 distance
    Args:
        points: tuple (y,x)
    Returns:
        [vx,vy,x,y]:
            vx,vy  = collinear vector
            x, y = point on the line
    """
    #TODO:https://stackoverflow.com/questions/47400174/what-does-opencv-fitline-do
    #TODO:https://en.wikipedia.org/wiki/Total_least_squares#Geometrical_interpretation

    count = len(points[0])

    E_x = np.sum(points[1]) / count
    E_y = np.sum(points[0]) / count
    E_x2 = np.sum(points[1] * points[1]) / count
    E_y2 = np.sum(points[0] * points[0]) / count
    E_xy = np.sum(points[1] * points[0]) / count

    Var_x = E_x2 - E_x * E_x
    Var_y = E_y2 - E_y * E_y
    Cov_xy = E_xy - E_x * E_y

    t = np.arctan2(2 * Cov_xy, Var_x - Var_y + np.sqrt( (Var_x - Var_y) * (Var_x - Var_y) + 4 * Cov_xy*Cov_xy ) )
    #robust simplification:
    #t = np.arctan2( 2 * Cov_xy, Var_x - Var_y ) / 2

    return [np.cos(t),np.sin(t),E_x,E_y]

def avgL2Distance(points, line):
    """
    Args:
        points: set of points (y,x)
        line: [vx,vy,x,y] directional vector + point on line
    Returns:
         normalized sum of L2 distances
    """
    num_points = len(points[0])

    v1 = np.array([ np.repeat(line[0],num_points),
                    np.repeat(line[1],num_points)])
    v2 = np.array([points[1]-line[2],points[0]-line[3]])

    #h = 2*A / ...
    return np.sum(np.abs(np.cross(v1.T,v2.T)) / np.linalg.norm(v1.T, axis=1)) / num_points

def filterPoints(points,x,y,max_dist):
    """
    Args:
        points: set of points (y,x)
        x,y: reference coordinate
        max_dist: maximal distance between reference and points allowed
    Returns:
        (y,x) subset which is within max_dist to reference point
    """
    #find candidates
    mask = np.sqrt(np.power((points[1]-x),2) + np.power((points[0]-y),2)) <= max_dist
    candidates = (points[0][mask==True],points[1][mask==True])

    #iterate over neighbors
    pre_x, pre_y = x, y
    curr_x, curr_y = x, y
    found_x = []
    found_y = []
    found = True
    
    while (found):
        found = False
        for next_y, next_x in zip(candidates[0],candidates[1]):
            if (curr_x == next_x) and (curr_y == next_y):
                continue
            if (pre_x == next_x) and (pre_y == next_y):
                continue
            if (np.abs(curr_x - next_x) <= 1) and (np.abs(curr_y - next_y) <= 1):
                found_x.append(next_x)
                found_y.append(next_y)
                pre_x, pre_y = curr_x, curr_y
                curr_x, curr_y = next_x, next_y
                found = True

    return (np.array(found_y),np.array(found_x))

def minDistPoint(points,x,y):
    """
    Args:
        points: set of points (y,x)
        x,y: reference point
    Returns:
        point with minimal L2 distance to reference
    """
    dist = np.sqrt(np.power((points[1]-x),2) + np.power((points[0]-y),2))
    idx = np.where(dist == np.amin(dist))
    return((points[0][idx][0],points[1][idx][0]))

def meanVector(points, x,y):
    """
    Args:
        points: set of points (y,x)
        x,y: defined vector origin
    Returns:
        mean unit vector [dx,dy]
    """
    count = len(points[0])

    E_x = np.sum(points[1]) / count
    E_y = np.sum(points[0]) / count

    vec = [E_x - x, E_y - y]
    unit_vec = vec / np.linalg.norm(vec)

    return unit_vec

def angleOfVectors(unit_vec1, unit_vec2):
    """
    Args:
        vec1,vec2: two unit vectors
    Returns:
        angle between both vectors
    """
    alpha = np.arccos(np.dot(unit_vec1, unit_vec2))
    if np.isnan(alpha):
        return 0
    else:
        return alpha

if __name__ == "__main__":
    #x = np.array([1,19,1,19,1,19,1,19,1,19,1,19])
    x = np.array([4,5,6,7,8,9,10,11,12,13,14,15,4,4,4])
    y = np.array([4,5,6,7,8,9,10,11,12,13,14,15,5,6,7])
    #y = np.array([10,10,10,10,10,10,10,10,10,10,10,10])
    
    print(filterPoints((y,x),4,4,5))

    #vec1 = meanVector((y,x),16,16)
    #vec2 = meanVector((y,x), 0,0)
    #alpha = angleOfVectors(vec1,vec2)
    #min = minDistPoint((y,x),0,0)