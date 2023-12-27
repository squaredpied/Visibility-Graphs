#!/usr/bin/python3

from matplotlib import pyplot as plt
from matplotlib import path
import numpy as np
import sys
import csv
import math
from typing import List

class Point:
    def __init__(self, x=0.0, y=0.0, id=0, poly_id=0):
        self.x = x
        self.y = y
        self.id=id
        self.poly_id=poly_id
    
    def dist(self, p):
        # Distance between point self and point p
        return math.sqrt((self.x - p.x)**2 + (self.y - p.y)**2)
    
    def numpy(self):
        # return the point (x, y) as a numpy array
        return np.array([self.x, self.y])
        
    def dist_line(self, l):
        # return the distance between point self an line l of type Segment.
        return np.linalg.norm(np.cross(l.p2.numpy() - l.p1.numpy(), l.p1.numpy() - self.numpy())) / np.linalg.norm(l.p2.numpy() - l.p1.numpy())

    def __str__(self):
        # returns point self as a string
        return "({}, {})".format(np.round(self.x, 2), np.round(self.y, 2))

    def dot(self, p):
        # Dot product
        return self.x * p.x + self.y*p.y

    def length(self):
        # returns modulus of point self
        return math.sqrt(self.x**2 + self.y**2)

    def vector(self, p):
        # creates a vector of type Point between point self and point p
        return Point(p.x - self.x, p.y - self.y)

    def unit(self):
        # makes the point self unitary if possible
        mag = self.length()
        if mag > 0:
            return Point(self.x/mag, self.y/mag)
        else:
            return Point(0, 0)

    def scale(self, sc):
        # multiplies point self by scalar sc
        return Point(self.x * sc, self.y * sc)

    def __add__(self, p):
        # add point self and point p component by component
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        # substracts point self and point p component by component
        return Point(self.x - p.x, self.y - p.y)

    def __truediv__(self, s):
        # divides point self by scalar s
        return Point(self.x / s, self.y / s)
    
    def __floordiv__(self, s):
        # integer division of point self by scalar s
        return Point(int(self.x / s), int(self.y / s))
    
    def __mul__(self, s):
        return Point(self.x * s, self.y * s)
    
    def __rmul__(self, s):
        return self.__mul__(s)
    
    def __eq__(self, __o: object) -> bool:
        if abs(self.x - __o.x) < 0.0001 and abs(self.y - __o.y) < 0.0001:
            return True
        return False 


def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) >= (B.y - A.y) * (C.x - A.x)

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

class Segment:
    def __init__(self, p1=Point(), p2=Point()):
        # A segment is defined by two Point objects
        self.p1 = p1
        self.p2 = p2

    @classmethod
    def point_angle_length(cls, p1=Point(), angle=0, length=1):
        # A segment can be initialized with a Point object, an angle, and a segment length.
        x2 = p1.x + math.cos(angle) * length
        y2 = p1.y + math.sin(angle) * length
        return cls(p1, Point(x2, y2))
        
    def intersect(self, s):
        # Return true if Segment self and Segment s intersect
        if ccw(self.p1, s.p1, s.p2) != ccw(self.p2, s.p1, s.p2) and ccw(self.p1, self.p2, s.p1) != ccw(self.p1, self.p2, s.p2):
            p = self.intersection_point(s)
            if p == self.p1 or p == self.p2:
                return False, None
            else:
                return True, p
        else:
            return False, None

    def intersection_point(self, line):
        # Returns the point in which line Segment self and line Segment s intersect
        xdiff = (self.p1.x - self.p2.x, line.p1.x - line.p2.x)
        ydiff = (self.p1.y - self.p2.y, line.p1.y - line.p2.y)

        div = det(xdiff, ydiff)
        if div == 0:
            print("Something went wrong!")
            return None

        d = (det((self.p1.x, self.p1.y), (self.p2.x, self.p2.y)), det((line.p1.x, line.p1.y), (line.p2.x, line.p2.y)))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Point(x, y)
    
    def anglefromhorizontal(self):
        angle= math.atan2((self.p2.y-self.p1.y),(self.p2.x-self.p1.x))
        if angle<0:
            angle+=2*math.pi
        return angle
    
    def __str__(self):
        return "[{}, {}]".format(self.p1, self.p2)


def load_vertices_from_file(filename: str):
    # list of vertices
    vertices = []
    current_id = 0
    with open(filename, newline='\n') as csvfile:
        v_data = csv.reader(csvfile, delimiter=",")
        next(v_data)
        for row in v_data:
            vertex = Point(float(row[1]), float(row[2]), current_id, int(row[0]))
            vertices.append(vertex)
            current_id += 1
    return vertices

def plot(vertices, edges=None, LineSegs=None):
    #plot the vertices
    for v in vertices:
        plt.plot(v.x, v.y, 'r+')

    #plot the edges in the environment
    for env in LineSegs:
        plt.plot([env.p1.x, env.p2.x],[env.p1.y, env.p2.y], 'k')
    
    #plot the visibility graph
    if edges!=None:
        for e in edges:
            plt.plot([vertices[e[0]].x, vertices[e[1]].x],
                 [vertices[e[0]].y, vertices[e[1]].y],
                 "g--")

    for i, v in enumerate(vertices):
        plt.text(v.x + 0.2, v.y, str(i))
    plt.axis('equal')
    plt.show()

def LineSegmentEnv(vertices):
    #Get the line segments (edges) in an environment
    pointcon=[]
    poly=-1
    LineSegments=[]
    for i in range(0,vertices[-1].id+1):
            if poly!=-1 and len(pointcon)!=0 and poly!=vertices[i].poly_id:
                linesegment=Segment(pointcon[-1], pointcon[0])
                LineSegments.append(linesegment)
            if poly!=vertices[i].poly_id:
                poly=vertices[i].poly_id
                pointcon=[]
                pointcon.append(vertices[i])
            else:
                linesegment=Segment(pointcon[-1],vertices[i])
                LineSegments.append(linesegment)
                pointcon.append(vertices[i])
    return LineSegments

def collinearitycheck(a, b, c):
    val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)
    if val == 0:
        # Collinear
        return 0
    return 1

def checkInside(polygon, point):
    #Ray casting algorithm
    n=len(polygon)
    count = 0
    for i in range(n):
        # Get an edge of the polygon
        j=(i+1)%n
        if ((polygon[i].y > point.y) != (polygon[j].y > point.y)) and \
           (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x):
            count += 1
 
    return count & 1

def isvisible(v:Point,vi:Point,S, vertices:List[Point]):
    #Check for colinearity with other vertices
    segments=LineSegmentEnv(vertices)
    for vertex in vertices:
        if vertex!=v and vertex!=vi:
            colinear=collinearitycheck(v,vi,vertex)
            if colinear==0:
                if (vertex.x>v.x and vertex.x<vi.x) or (vertex.y>v.y and vertex.y<vi.y) or (vertex.x>vi.x and vertex.x<v.x) or (vertex.y>vi.y and vertex.y<v.y):
                    return False

    #Check for vertices of the same polygon id   
    if v.poly_id == vi.poly_id:
        Poly=[]
        for seg in segments:
            if ((seg.p1==vi) and (seg.p2==v)) or ((seg.p1==v) and (seg.p2==vi)):
                return True

            if (seg.p1.poly_id==v.poly_id):
                Poly.append(seg.p1)
        mid_point=Point((v.x+vi.x)*0.5,(v.y+vi.y)*0.5)
        return not checkInside(Poly,mid_point)

    #Return True if S is empty
    if len(S)==0:
        return True
    
    #Check if edge (v, vi) intersects the first edge on S
    current_seg=Segment(v,vi)
    first_seg=list(S.keys())[0]
    bin,_=current_seg.intersect(first_seg)
    if bin==True:
        if (first_seg.p1!=vi) or (first_seg.p2!=vi) or (first_seg.p1!=v) or (first_seg.p2!=v):
            return False
    else:
        return True

def RotationalSweepAlg(v, vertices: List[Point]):
    # Initialize empty lists and dictionaries
    subset_v = []
    S = {}
    edges = []
    maximum_x = float('-inf')
    new_vertices = []

    # Create line segments and update maximum_x
    for vertex in vertices:
        if vertex != v:
            edges.append(Segment(v, vertex))
            new_vertices.append(vertex)
        maximum_x = vertex.x if vertex.x > maximum_x else maximum_x

    # Calculate angles of line segments from the horizontal axis
    epsilon = [seg.anglefromhorizontal() for seg in edges]

    # Sort vertices and angles in ascending order of angles
    indexes = sorted(range(len(epsilon)), key=epsilon.__getitem__)
    new_vertices = list(map(new_vertices.__getitem__, indexes))
    epsilon = list(map(epsilon.__getitem__, indexes))

    # Create a horizontal line slightly to the right of the vertices
    horizontal_line = Segment(v, Point(x=maximum_x + 1, y=v.y))

    # Initialize the environment (line segments)
    env = LineSegmentEnv(vertices)

    # Process intersection with the horizontal line and store distances in S
    for i in env:
        cond, point1 = horizontal_line.intersect(i)
        if cond == True:
            S[i] = point1.dist(v)

    # Sort S based on distances
    S = dict(sorted(S.items(), key=lambda x: x[1]))

    # Main loop for each angle and corresponding vertex
    for angle, vertex in zip(epsilon, new_vertices):
        # Check visibility and add to subset_v if visible
        if isvisible(v, vertex, S, vertices) == True:
            subset_v.append((v.id, vertex.id))

        # Update S based on intersections with edges connected to the current vertex
        for edge in env:
            if (edge.p1 == vertex) or (edge.p2 == vertex):
                if S.get(edge) is None:
                    # Create a line segment slightly offset from the angle and find intersection
                    line_seg = Segment.point_angle_length(v, angle + 0.00000001, 99999)
                    cond, point1 = line_seg.intersect(edge)
                    if cond == True:
                        S[edge] = point1.dist(v)
                else:
                    # If the edge is already in S, remove it
                    del S[edge]

        # Sort S based on distances
        S = dict(sorted(S.items(), key=lambda x: x[1]))

    # Return the subset of visible vertices
    return subset_v

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\n ${} file_name.csv".format(sys.argv[0]))
    else:
        # execute visibility graph algorithm
        vertices=load_vertices_from_file(sys.argv[1])

        LineSegs=LineSegmentEnv(vertices)
        visibility_graph=[]
        
        #Implement RPS for all vertices
        for vertex in vertices:
            vert=RotationalSweepAlg(vertex,vertices)
            visibility_graph=visibility_graph+vert
        
        #remove duplicate edges
        for x in visibility_graph:
            for y in visibility_graph:
                if (x[0]==y[1] and x[1]==y[0]):
                    visibility_graph.remove((y[0],y[1]))

        print(visibility_graph)
        plot(vertices, visibility_graph, LineSegs=LineSegs)