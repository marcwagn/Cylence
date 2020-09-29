import numpy as np
import cv2
import networkx as nx
import itertools

from skimage.morphology import skeletonize
from skimage.draw import line

from scipy import signal

from .leastSquareAnalysis import *

class ConnCompAnalysis():

    def __init__(self, pxl, stats):
        """
        Args:
            cnt: representation of the contour
            img: original binary image (contour is included)
        """
        #stats
        self.x, self.y, self.width, self.height, self.size  = stats
    
        #data
        self.bin_img = np.zeros((self.height,self.width), dtype=np.uint8)
        self.bin_img[pxl[0], pxl[1]] = 1

        self.skltn = skeletonize(self.bin_img,method='lee')
        self.skltn_pxl = np.where(self.skltn == 1)

        #estimate filament width
        self.fila_width = self.estimateWidthFilaments()

        #create Graph
        self.G = nx.MultiGraph()

        #set hyperparameter resolveNode/resolveNodePair
        self.max_dist = 15
        self.max_angle = 90
        self.max_mean_dev = 5.0

    def estimateWidthFilaments(self, step=1):
        """ 
        Args:
            bin_img: binary image of filaments
            step: step size (e.g. 2 = every secound pixel)
        Returns:
            maximal width of filaments
        """
        #init kernel
        size = 3
        kernel = self.createCircleKernel(size)
        kernel_radius = size // 2
        kernel_value = np.sum(kernel)

        #iterate over pixels in skeleton
        for y_kernel, x_kernel in zip (self.skltn_pxl[0], self.skltn_pxl[1]):

            while(True):
                #check if kernel fits in image
                if(
                    (y_kernel >= self.height - kernel_radius) or (y_kernel < kernel_radius) or 
                    (x_kernel >= self.width - kernel_radius) or (x_kernel < kernel_radius)
                  ):
                  break
                #calculate match score for given kernel
                featureScore = np.sum(self.bin_img[ y_kernel-kernel_radius:y_kernel+kernel_radius+1,
                                                    x_kernel-kernel_radius:x_kernel+kernel_radius+1] * kernel)

                #check if match score is maximal
                if featureScore == kernel_value :
                    #if yes: enlarge kernel
                    size += 2
                    kernel = self.createCircleKernel(size)
                    kernel_radius = size // 2
                    kernel_value = np.sum(kernel)
                else:
                    #if no: go to next pixel
                    break
        
        return size

    def checkConnCompSize(self):
        """
        Args:
            self.fila_width: extimated filament width
            self.size: number pixel in conn comp.
        Returns:
            True: if size matches criteria
        """
        if self.fila_width > 30:
            return False
        if self.fila_width <= 5:
            return False
        if self.size < (self.fila_width * 2 * self.fila_width):
            return False
        if self.size < (200):
            return False

        return True

    def createGraph(self):
        """
        Args: 
            self.skltn: 1px skeleton of filament
        Returns:
            self.G: graph representation of the skeleton
        """
        kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
        tmp = signal.convolve2d(self.skltn,kernel,boundary='fill', mode='same')

        #detect endpoints
        end_map = np.where(tmp == 2,1,0).astype(np.uint8)
        end_map = np.bitwise_and(end_map,self.skltn)
        endpoints = np.where(end_map == 1)

        for y, x in zip(endpoints[0], endpoints[1]):
            self.G.add_node(str(y)+"_"+str(x), 
                            points=(np.array([y]),np.array([x])))

        #detect intersections
        inter_map = np.where(tmp > 3,1,0).astype(np.uint8)
        inter_map = np.bitwise_and(inter_map,self.skltn)
        intersections = np.where(inter_map == 1)

        for y, x in zip(intersections[0], intersections[1]):
            self.G.add_node(str(y)+"_"+str(x), 
                            points=(np.array([y]),np.array([x])))

            #connect neighbouring intersection
            for y_neig in [-1,0,1]:
                for x_neig in [-1,0,1]:
                    #prevent self loops
                    if x_neig == 0 and y_neig == 0:
                        continue
                    if self.G.has_node(str(y+y_neig)+"_"+str(x+x_neig)):
                        endpoint_idx_1 = str(y+y_neig)+"_"+str(x+x_neig)
                        endpoint_idx_2 = str(y)+"_"+str(x)
                        self.G.add_edge(endpoint_idx_1,
                                        endpoint_idx_2,
                                        points = None,
                                        length=0,
                                        endpoints = [(endpoint_idx_1,None), (endpoint_idx_2,None)]
                                        )

        #detect edges
        edges_map = (self.skltn - end_map) - inter_map
        #detect endpoints edges
        tmp = signal.convolve2d(edges_map,kernel,boundary='fill', mode='same')
        edges_end_map = np.where(tmp == 2,1,0).astype(np.uint8)
        #detect edges pixel
        edges = cv2.connectedComponents(edges_map, cv2.CV_32S, connectivity = 8)

        for idx in range(1,edges[0]):
            #find endpoints of edge
            endpoint_idx_1 = None
            endpoint_idx_2 = None
            endpoint_1 = None
            endpoint_2 = None

            edge = np.where(edges[1] == idx,1,0).astype(np.uint8)
            edge_pxl = np.where(edge == 1)

            #case 1: edge has just one pixel
            if len(edge_pxl[0]) == 1:
                node_idx = []
                for y_neig in [-1,0,1]:
                    for x_neig in [-1,0,1]:
                        tmp_idx = str(edge_pxl[0][0]+y_neig)+"_"+str(edge_pxl[1][0]+x_neig)
                        if self.G.has_node(tmp_idx):
                            node_idx.append(tmp_idx)
                endpoint_idx_1 = node_idx[0]
                endpoint_1 = (edge_pxl[0][0],edge_pxl[1][0])
                endpoint_idx_2 = node_idx[1]
                endpoint_2 = (edge_pxl[0][0],edge_pxl[1][0])
            #case 2: edge has more than one pixel
            else:
                edge_end = np.bitwise_and(edge,edges_end_map)
                edge_end_pxl = np.where(edge_end == 1)
                
                #check for circles 
                if (edge_end_pxl[0].size == 0 or edge_end_pxl[1].size == 0):
                    continue

                for y_neig in [-1,0,1]:
                    for x_neig in [-1,0,1]:
                        tmp_idx_1 = str(edge_end_pxl[0][0]+y_neig)+"_"+str(edge_end_pxl[1][0]+x_neig)
                        tmp_idx_2 = str(edge_end_pxl[0][1]+y_neig)+"_"+str(edge_end_pxl[1][1]+x_neig)

                        if self.G.has_node(tmp_idx_1):
                            endpoint_idx_1 = tmp_idx_1
                            endpoint_1 = (edge_end_pxl[0][0],edge_end_pxl[1][0])
                        if self.G.has_node(tmp_idx_2):
                            endpoint_idx_2 = tmp_idx_2
                            endpoint_2 = (edge_end_pxl[0][1],edge_end_pxl[1][1])
                
            #check if we found two endpoints
            if (endpoint_idx_1 != None) and (endpoint_idx_2 != None):
                #add edge
                self.G.add_edge(endpoint_idx_1,
                                endpoint_idx_2,
                                points = edge_pxl,
                                length=len(edge_pxl[0]),
                                endpoints = [(endpoint_idx_1,endpoint_1),(endpoint_idx_2,endpoint_2)]
                                )

    def visualizeGraph(self, img, bold = True):
        """
        Args:
            img: rgb image 
        Returns:
            rbg image with colored graph
        """
        if bold:
            size_vec = [-1,0,1]
        else:
            size_vec = [0]

        #draw edges
        h, w, _ = img.shape
        for _,_,edata in self.G.edges(data=True):
            if edata['points'] != None:
                color = self.randomColor()
                for y_neig in size_vec:
                    for x_neig in size_vec:
                        y = np.clip(edata['points'][0] + y_neig,0,h-1)
                        x = np.clip(edata['points'][1] + x_neig, 0, w-1)
                        img[y,x,:] = color
                        y,x = edata['endpoints'][0][1]
                        img[y,x,:] = [255,255,255]
                        y,x = edata['endpoints'][1][1]
                        img[y,x,:] = [255,255,255]
        #draw nodes
        for _, edata in self.G.nodes(data=True):
            if edata['points'] != None:
                color = [0,0,0]
                for y_neig in size_vec:
                    for x_neig in size_vec:
                        y = np.clip(edata['points'][0] + y_neig,0,h-1)
                        x = np.clip(edata['points'][1] + x_neig, 0, w-1)
                        img[y,x,:] = color
 
    def pruneGraph(self):
        """
        Args:
            self.G: Graph
            self.fila_width
        Returns:
            removes 
        """
        #step 1: prune graph
        rmve_lst = []
        check_lst = []
        for u,v,edata in self.G.edges(data=True):
            #case 1: prune u
            if (self.G.degree[u] == 1 and 
                    self.G.degree[v] > 1 and
                    edata['length'] < self.fila_width):
                rmve_lst.append(u)
                check_lst.append(v)
            #case 2: prune v
            if (self.G.degree[u] > 1 and 
                    self.G.degree[v] == 1 and
                    edata['length'] < self.fila_width):
                rmve_lst.append(v)
                check_lst.append(u)
        #remove nodes
        for node in rmve_lst:
            self.G.remove_node(node)

        #check if inner node can be removed
        for node in check_lst:
            #node already removed
            if not self.G.has_node(node):
                continue
            if self.G.degree[node] == 2 and len(list(self.G.neighbors(node))) == 2:
                neigh1, neigh2 = list(self.G.neighbors(node))
                self.joinEdges(node,neigh1,0,None,node, neigh2, 0, None)
                self.G.remove_node(node)

    def joinEdges(self, u1, v1, k1, end1, u2, v2, k2, end2):
        """
        Args:
            edge1: (node1,node,key1)
            edge2: (node2,node,key2)
            node: shared node of edge1 and edge2
            endX: give endpoints which should be joined
        Return:
            creates joined edge and removes edge1 and edge2
        """
        #circle detected
        if {u1,v1,k1} == {u2,v2,k2}:
            self.G.remove_edge(u1,v1, key=k1)
            return

        #get points and endpoints
        endpoints = []
        if end1 == None or end2 == None:
            if u1 == u2:
                y = self.G.nodes[u1]['points'][0]
                x = self.G.nodes[u2]['points'][1]
            else:
                y = np.concatenate((self.G.nodes[u1]['points'][0],
                                    self.G.nodes[u2]['points'][0]))
                x = np.concatenate((self.G.nodes[u1]['points'][1],
                                    self.G.nodes[u2]['points'][1]))
        else:
            y, x = line(end1[0],end1[1],end2[0],end2[1])

        if self.G[u1][v1][k1]['points'] != None:
                y = np.concatenate((y, self.G[u1][v1][k1]['points'][0]))
                x = np.concatenate((x, self.G[u1][v1][k1]['points'][1]))
                for e in self.G[u1][v1][k1]['endpoints']:
                    if e[0] == v1 and e[1] != end1:
                        endpoints.append(e)
        else:
            endpoints.append((v1,(self.G.nodes[u1]['points'][0][0],self.G.nodes[u1]['points'][1][0])))
    
        if self.G[u2][v2][k2]['points'] != None:
            y = np.concatenate((y, self.G[u2][v2][k2]['points'][0]))
            x = np.concatenate((x, self.G[u2][v2][k2]['points'][1]))
            for e in self.G[u2][v2][k2]['endpoints']:
                if e[0] == v2 and e[1] != end2:
                    endpoints.append(e)
        else:
            endpoints.append((v2,(self.G.nodes[u2]['points'][0][0],self.G.nodes[u2]['points'][1][0])))
            
        points = (y,x)
        length = len(x)
            
        #add edge
        self.G.add_edge(v1,
                        v2,
                        points = points,
                        length = length,
                        endpoints = endpoints)
        #remove edges
        self.G.remove_edge(u1,v1, key=k1)
        self.G.remove_edge(u2,v2, key=k2)

    def createEndnode(self, node, neig, key, end):
        """
        """
        #add new node
        new_node = "{}_{}".format(end[0],end[1])
        self.G.add_node(new_node,
                        points=(np.array([end[0]]),np.array([end[1]])))

        #add new edge
        #remove endpoint
        y,x = self.G[node][neig][key]['points']
        mask = ~np.logical_and(y == end[0], x == end[1])
        points = (y[mask],x[mask])
        length = len(points[0])

        #create endpoint list
        endpoints = []
        for e in self.G[node][neig][key]['endpoints']:
            if e[0] == neig:
                endpoints.append(e)
        #new endpoint
        tmp_end = minDistPoint(points,end[1],end[0])
        endpoints.append((new_node,tmp_end))

        self.G.add_edge(new_node,
                        neig,
                        points = points,
                        length = length,
                        endpoints = endpoints)

        self.G.remove_edge(node, neig,key=key)

        return new_node

    def createHyperNodes(self, alphaHyper = 3):
        """
        Args:
        Returns:
        """
        hyper_idx = 0
        found = True
        u,v = None, None

        while(found):
            found = False
            for u,v,key,edata in self.G.edges(keys=True, data=True):
                if (edata['length'] < alphaHyper * self.fila_width):
                    #get points
                    if u != v:
                        y = np.concatenate((self.G.nodes[u]['points'][0],
                                            self.G.nodes[v]['points'][0]))
                        x = np.concatenate((self.G.nodes[u]['points'][1],
                                            self.G.nodes[v]['points'][1]))
                    else:
                        y = self.G.nodes[u]['points'][0]
                        x = self.G.nodes[u]['points'][1]
                    
                    if edata['points'] != None:
                        y = np.concatenate((y, edata['points'][0]))
                        x = np.concatenate((x, edata['points'][1]))
                    
                    points = (y,x)

                    #add hypernode
                    hyper_node = "hyper_"+str(hyper_idx)
                    self.G.add_node(hyper_node,
                                    points = points)

                    #add edges from u to hypernode and self loops
                    for _ , n, n_key, n_edata in list(self.G.edges(u,keys=True, data=True)):
                        #create new self loop
                        if n == v and n_key != key:
                            endpoints = [(hyper_node,n_edata['endpoints'][0][1]),
                                        (hyper_node,n_edata['endpoints'][1][1])]
                            self.G.add_edge(hyper_node,hyper_node,
                                            points = n_edata['points'],
                                            length = n_edata['length'],
                                            endpoints = endpoints)
                        #copy self loop from u
                        elif n != v and n==u:
                            endpoints = [(hyper_node,n_edata['endpoints'][0][1]),
                                        (hyper_node,n_edata['endpoints'][1][1])]
                            self.G.add_edge(hyper_node,hyper_node,
                                            points = n_edata['points'],
                                            length = n_edata['length'],
                                            endpoints = endpoints)
                        #create new edge from n to hypernode
                        else:
                            if n_edata['endpoints'][0][0] == n:
                                endpoints = [(n,n_edata['endpoints'][0][1]),
                                             (hyper_node,n_edata['endpoints'][1][1])]
                            else:
                                endpoints = [(hyper_node,n_edata['endpoints'][0][1]),
                                             (n,n_edata['endpoints'][1][1])]
                            self.G.add_edge(n,hyper_node,
                                            points = n_edata['points'],
                                            length = n_edata['length'],
                                            endpoints = endpoints)

                    if u != v:
                        #add edges from v to hyper node
                        for _ , n, n_edata in self.G.edges(v, data=True):
                            if n == u:
                                continue
                            #self loops from v
                            elif n != u and n==v:
                                endpoints = [(hyper_node,n_edata['endpoints'][0][1]),
                                             (hyper_node,n_edata['endpoints'][1][1])]
                                self.G.add_edge(hyper_node,hyper_node,
                                                points = n_edata['points'],
                                                length = n_edata['length'],
                                                endpoints = endpoints)
                            #new edges
                            else:
                                if n_edata['endpoints'][0][0] == n:
                                    endpoints = [(n,n_edata['endpoints'][0][1]),
                                                 (hyper_node,n_edata['endpoints'][1][1])]
                                else:
                                    endpoints = [(hyper_node,n_edata['endpoints'][0][1]),
                                                 (n,n_edata['endpoints'][1][1])]
                                self.G.add_edge(n,hyper_node,
                                                points = n_edata['points'],
                                                length = n_edata['length'],
                                                endpoints = endpoints)
                    
                    #remove old nodes
                    self.G.remove_node(u)
                    if u != v:
                        self.G.remove_node(v)
                    hyper_idx += 1
                    found = True
                    
                    break

    def resolveNode(self, node):
        """
        Args:
            node: given node
        Returns:
            eliminates all edge pairs
        """
        while (self.G.degree[node] > 0):
            #generate list of endpoints
            edge_lst = []
            for u, v, key, data in self.G.edges(nbunch=node, keys=True, data=True):
                for endpoint in data['endpoints']:
                    if endpoint[0] == node:
                        edge_lst.append((u,v,key,endpoint[1]))
            
            #just one edge left
            if (self.G.degree[node] == 1):
                self.createEndnode(edge_lst[0][0], edge_lst[0][1], edge_lst[0][2], edge_lst[0][3])
                break
            
            #more than one edge left
            edge_comb_lst = list(itertools.combinations(edge_lst, 2))
        
            #calculate distance for each line combination
            dist_lst = []
            for (u1,v1,k1,end1), (u2,v2,k2,end2) in edge_comb_lst:
                #get points with given distance
                points1 = filterPoints(self.G[u1][v1][k1]['points'],end1[1],end1[0], self.max_dist)
                points2 = filterPoints(self.G[u2][v2][k2]['points'],end2[1],end2[0], self.max_dist)
                points = (np.concatenate((points1[0],points2[0])),np.concatenate((points1[1],points2[1])))
                
                #fit line + sum distances
                line = fitLine(points)
                avgL2dist = avgL2Distance(points,line)

                unit_vec1 = meanVector(points1,end1[1],end1[0])
                unit_vec2 = meanVector(points2,end2[1],end2[0])
                angle = np.radians(180) - angleOfVectors(unit_vec1,unit_vec2)

                dist_lst.append((angle, avgL2dist))
            
            sum_dist_lst = list(map(sum, dist_lst))
            min_idx = sum_dist_lst.index(min(sum_dist_lst))

            edge1, edge2 = edge_comb_lst[min_idx]
            if (dist_lst[min_idx][0] < np.radians(self.max_angle)) and (dist_lst[min_idx][1] < self.max_mean_dev):
                self.joinEdges(edge1[0],edge1[1],edge1[2],edge1[3],edge2[0],edge2[1],edge2[2], edge2[3])
            else:
                if edge1[0] == edge2[0] and edge1[1] == edge2[1] and edge1[2] == edge2[2]:
                    new_node = self.createEndnode(edge1[0],edge1[1],edge1[2],edge1[3])
                    self.createEndnode(edge2[0],new_node,edge2[2],edge2[3])
                else:
                    self.createEndnode(edge1[0],edge1[1],edge1[2],edge1[3])
                    self.createEndnode(edge2[0],edge2[1],edge2[2],edge2[3])
            
        self.G.remove_node(node)

    def resolveNodePair(self, node1, node2):
        """
        Args:
        Returns:
        """
        crossing = False
        while (True):
            #generate list of endpoints
            edge_lst = []
            for u, v, key, data in list(self.G.edges(nbunch=node1, keys=True, data=True)):
                if v == node2:
                    continue
                for endpoint in data['endpoints']:
                    if endpoint[0] == node1:
                        edge_lst.append((u,v,key,endpoint[1]))

            for u, v, key, data in list(self.G.edges(nbunch=node2, keys=True, data=True)):
                if v == node1:
                    continue
                for endpoint in data['endpoints']:
                    if endpoint[0] == node2:
                        edge_lst.append((u,v,key,endpoint[1]))

            #break while True loop
            if (len(edge_lst) <= 1):
                if len(edge_lst) == 1:
                    self.createEndnode(edge_lst[0][0], edge_lst[0][1], edge_lst[0][2], edge_lst[0][3])
                #check for hidden filament
                if (crossing == False):
                    for u, v, key, data in list(self.G.edges(nbunch=node1, keys=True, data=True)):
                        for endpoint in data['endpoints']:
                            if endpoint[0] == u:
                                self.createEndnode(u,v,key,endpoint[1])
                    for u, v, key, data in list(self.G.edges(nbunch=node2, keys=True, data=True)):
                        for endpoint in data['endpoints']:
                            if endpoint[0] == u:
                                self.createEndnode(u,v,key,endpoint[1])
                break
            
            #more than one edge left
            edge_comb_lst = list(itertools.combinations(edge_lst, 2))
        
            #calculate distance for each line combination
            dist_lst = []
            for (u1,v1,k1,end1), (u2,v2,k2,end2) in edge_comb_lst:
                #get points with given distance
                points1 = filterPoints(self.G[u1][v1][k1]['points'],end1[1],end1[0], self.max_dist)
                points2 = filterPoints(self.G[u2][v2][k2]['points'],end2[1],end2[0], self.max_dist)
                points = (np.concatenate((points1[0],points2[0])),np.concatenate((points1[1],points2[1])))
                
                #fit line + sum distances
                line = fitLine(points)
                avgL2dist = avgL2Distance(points,line)

                unit_vec1 = meanVector(points1,end1[1],end1[0])
                unit_vec2 = meanVector(points2,end2[1],end2[0])
                angle = np.radians(180) - angleOfVectors(unit_vec1,unit_vec2)

                dist_lst.append((angle, avgL2dist))
            
            sum_dist_lst = list(map(sum, dist_lst))
            min_idx = sum_dist_lst.index(min(sum_dist_lst))
            
            edge1, edge2 = edge_comb_lst[min_idx]
            if (dist_lst[min_idx][0] < np.radians(self.max_angle)) and (dist_lst[min_idx][1] < self.max_mean_dev):
                self.joinEdges(edge1[0],edge1[1],edge1[2],edge1[3],edge2[0],edge2[1],edge2[2],edge2[3])
                if edge1[0] != edge2[0]:
                    crossing = True
            else:
                if edge1[0] == edge2[0] and edge1[1] == edge2[1] and edge1[2] == edge2[2]:
                    new_node = self.createEndnode(edge1[0],edge1[1],edge1[2],edge1[3])
                    self.createEndnode(edge2[0],new_node,edge2[2],edge2[3])
                else:
                    self.createEndnode(edge1[0],edge1[1],edge1[2],edge1[3])
                    self.createEndnode(edge2[0],edge2[1],edge2[2],edge2[3])

        self.G.remove_node(node1)
        self.G.remove_node(node2)

    def resolveEvenHyperNodes(self):
        """
        Args:
            self.G: Graph with hypernodes
        Returns:
            self.G: resolves all nodes with even degree
        """
        for node in list(self.G.nodes):
            if self.G.degree[node] % 2 == 0:
                self.resolveNode(node)
    
    def resolveOddNodes(self):
        """
        Args:
            self.G: Graph with hypernodes
            max_dist: maximal L2 distance of used points for linear fit
        Returns:
            self.G: without hidden endpoints 
        """
        for node in list(self.G.nodes):
            #node already removed
            if not self.G.has_node(node):
                continue
            if (self.G.degree[node] > 1) and (self.G.degree[node] % 2 == 1):
                #count neighbors with odd degree
                odd_neighbor_lst = []
                for neighbor in self.G.neighbors(node):
                    if ( self.G.degree[neighbor] > 1  and 
                         self.G.degree[neighbor] % 2 == 1 and 
                         node != neighbor ):
                        odd_neighbor_lst.append(neighbor)

                #case 1: no neighbors
                if len(odd_neighbor_lst) == 0:
                    self.resolveNode(node)

    def resolveOddNodes_advanced(self):
        """
        Args:
            self.G: Graph with hypernodes
            max_dist: maximal L2 distance of used points for linear fit
        Returns:
            self.G: without odd Nodes 
        """
        for node in list(self.G.nodes):
            #node already removed
            if not self.G.has_node(node):
                continue
            if (self.G.degree[node] > 1) and (self.G.degree[node] % 2 == 1):
                #count neighbors with odd degree
                odd_neighbor_lst = []
                for neighbor in self.G.neighbors(node):
                    if ( self.G.degree[neighbor] > 1  and 
                         self.G.degree[neighbor] % 2 == 1 and 
                         node != neighbor ):
                        odd_neighbor_lst.append(neighbor)

                #case 1: no neighbors
                if len(odd_neighbor_lst) == 0:
                    self.resolveNode(node)
                #case 2: exactly 1 neighbor
                elif len(odd_neighbor_lst) == 1:
                    self.resolveNodePair(node, odd_neighbor_lst[0])
                #case 3: >1 neighbors
                else:
                    min_neigh, min_dist = odd_neighbor_lst[0], np.sqrt(self.width**2 + self.height**2)

                    for odd_neighbor in odd_neighbor_lst:
                        for key in self.G[node][odd_neighbor]:
                            if self.G[node][odd_neighbor][key]['length'] < min_dist:
                                min_neigh, min_dist = odd_neighbor, self.G[node][odd_neighbor][key]['length']
                    
                    self.resolveNodePair(node, min_neigh)

    def removeUnresolved(self):
        """
        Args:
            self.G
        Returns:
            removes all connected components with more than 2 nodes
        """
        for comp in list(nx.connected_components(self.G)):
            if len(comp) > 2:
                for node in comp:
                    self.G.remove_node(node)

    def checkResolvedGraph(self):
        """
        Args:
            self.G
        Returns:
            True: when all nodes in G have degree 1
            False: otherwise
        """
        check = True
        for node in self.G.nodes:
            if self.G.degree[node] != 1:
                check = False
        return check

    def quantifyParameter(self, infection_map, area=25, thres=10, minLen=20):
        """
        Args:
            infection_map: binary feature map of infection (1 == infection)
            area (+ fila width): defines diameter of the search space
            thres: minimal number of infected pixel
        Returns:
            number of filaments
            number of infected filaments

        """
        #radius quadratic kernel
        r = (self.fila_width + area) // 2
        
        countFilament = 0
        countInfected = 0

        for _,_,_,data in self.G.edges(keys=True,data=True):
            #count filaments
            if data['length'] >= minLen:
                countFilament += 1
            else:
                continue
            #count infected filaments
            check_infection = False
            for endpoint in data['endpoints']:
                end_y,end_x = endpoint[1][0] + self.y, endpoint[1][1] + self.x
                inf_height, inf_width = infection_map.shape
                
                ymin = max(end_y-r,0)
                ymax = min(end_y+r,inf_height)
                xmin = max(end_x-r,0)
                xmax = min(end_x+r,inf_width)

                if np.sum(infection_map[ymin:ymax,xmin:xmax]) > thres:
                    check_infection = True
    
            if check_infection == True:
                countInfected += 1
        
        return (countFilament, countInfected)

    ### 
    ### helper functions
    ### 

    def createCircleKernel(self, size):
        """
        Args:
            size: size of quadratic kernel [size x size]
        Returns:
            binary 2D numpy array with a centered circle d=size 
        """
        #size must be odd to have center point
        if (size % 2 == 0):
            raise ValueError("Kernel size must be odd!")
        #find radius of maximal circle without center point
        radius = size // 2
        #set grid
        y,x = np.ogrid[-radius:size-radius, -radius:size-radius]
        #create boolean mask of the circle
        mask = x*x + y*y <= radius*radius
        #create kernel based on boolean mask
        kernel = np.zeros((size, size))
        kernel[mask] = 1

        return kernel

    def randomColor(self):
        """
        Returns:
            random color vector
        """
        return list(np.random.choice(range(50,256), size=3))
