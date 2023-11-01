import csv
from matplotlib import pyplot as plt
import numpy as np
import math
from typing import List


class Vertex:
    """
    Vertex class defined by x and y coordinate.
    """
    # constructor or initializer of vertex class

    def __init__(self, x=0, y=0, id=None):
        self.x = x
        self.y = y
        self.id = id
        self.neighbours = []

    def dist(self, p: "Vertex"):
        """
        Return distance between vertices
         Parameters:
            p: input vertex to calculate distance to.
         Returns:
            Distance to vertex from this vertex object
        """

        return math.sqrt((self.x - p.x)**2 + (self.y - p.y)**2)

    # method to define print() function of object vertex
    def __str__(self):
        return "({}, {})".format(np.round(self.x, 2), np.round(self.y, 2))

    # method to define print() function of list[] of object vertex
    def __repr__(self):
        return "({}, {})".format(np.round(self.x, 2), np.round(self.y, 2))


def plot(vertices, edges, path=None):

    for v in vertices:
        plt.plot(v.x, v.y, 'r+')

    for e in edges:
        plt.plot([vertices[e[0]].x, vertices[e[1]].x],
                 [vertices[e[0]].y, vertices[e[1]].y],
                 "g--")
    if path != None:
        points=np.zeros((len(path),2))
        for i in range(len(path)):
            points[i,0]=vertices[path[i]].x
            points[i,1]=vertices[path[i]].y
        plt.plot(points[:,0],points[:,1],'r')
    for i, v in enumerate(vertices):
        plt.text(v.x + 0.2, v.y, str(i))
    plt.axis('equal')



def load_vertices_from_file(filename: str):
    # list of vertices
    vertices: List[Vertex] = []
    current_id = 0
    with open(filename, newline='\n') as csvfile:
        v_data = csv.reader(csvfile, delimiter=",")
        next(v_data)
        for row in v_data:
            vertex = Vertex(float(row[1]), float(row[2]), id=current_id)
            vertices.append(vertex)
            current_id += 1
    return vertices


def load_edges_from_file(filename: str):
    edges = []
    with open(filename, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            edges.append((int(row[0]), int(row[1])))
    return edges

"""
print("vertices from file")
# save each "vertex" in vertices list
vertices = load_vertices_from_file("./env_0.csv")
for elem in vertices:
    print(elem)

print("\ndistance from vertex 0 to vertex 1")
print(vertices[0].dist(vertices[1]))

print("\nedges from file")
# save each edge in "edges" list
edges = load_edges_from_file("./visibility_graph_env_0.csv")
for elem in edges:
    print(elem)

plot(vertices, edges)
plt.show()
plt.axis('equal')
"""