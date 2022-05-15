import numpy as np
import sortedcontainers
import globalvar
from TransformPathToTrajectory import TransformPathToTrajectory
from main_unstructure import checkObj_linev, checkObj_point, CreateVehiclePolygon, distance, \
    GenerateStaticObstacles_unstructured, VisualizeStaticResults, VisualizeDynamicResults


def IsVertexValid(pt):
    for obstacle in globalvar.obstacles_[0]:
        obs = np.vstack((obstacle.x, obstacle.y))
        if checkObj_point(pt, obs):
            return False
    return True


# check feasible edge when produced
def IsLineValid(p1, p2):  # p1(x,y),p2(x,y) represent a vector p1----->p2
    theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    V1 = CreateVehiclePolygon(x=p1[0], y=p1[1], theta=theta)
    V2 = CreateVehiclePolygon(x=p2[0], y=p2[1], theta=theta)
    X = np.hstack((V1.x, V2.x))
    Y = np.hstack((V1.y, V2.y))
    min_x_index = np.argmin(X)
    max_x_index = np.argmax(X)
    min_y_index = np.argmin(Y)
    max_y_index = np.argmax(Y)
    min_x_v = np.min(X)
    max_x_v = np.max(X)
    min_y_v = np.min(Y)
    max_y_v = np.max(Y)
    P1 = np.array([X[min_x_index], Y[min_x_index]])
    P2 = np.array([X[max_x_index], Y[max_x_index]])
    P3 = np.array([X[min_y_index], Y[min_y_index]])
    P4 = np.array([X[max_y_index], Y[max_y_index]])
    for obstacle in globalvar.obstacles_[0]:

        obs = np.vstack((obstacle.x, obstacle.y))
        min_x_obs = np.min(obstacle.x)
        max_x_obs = np.max(obstacle.x)
        min_y_obs = np.min(obstacle.y)
        max_y_obs = np.max(obstacle.y)
        if (min_x_obs >= min_x_v and max_x_obs <= max_x_v and min_y_obs >= min_y_v and max_y_obs <= max_y_v) or \
                (min_x_obs <= min_x_v and max_x_obs >= max_x_v and min_y_obs <= min_y_v and max_y_obs >= max_y_v):
            return False
        if checkObj_linev(P1, P4, obs) or checkObj_linev(P4, P2, obs) or checkObj_linev(P2, P3, obs) or checkObj_linev(
                P3, P1, obs):
            return False

    return True


def getvertex(k):
    vertex = np.array([globalvar.vehicle_TPBV_.x0, globalvar.vehicle_TPBV_.y0])
    vertex = np.vstack((vertex, [globalvar.vehicle_TPBV_.xtf, globalvar.vehicle_TPBV_.ytf]))
    lx = globalvar.planning_scale_.xmin
    ux = globalvar.planning_scale_.xmax
    ly = globalvar.planning_scale_.ymin
    uy = globalvar.planning_scale_.ymax

    while vertex.shape[0] < k + 2:
        x = (ux - lx) * np.random.rand() + lx
        y = (uy - ly) * np.random.rand() + ly
        pt = np.array([x, y])
        if not IsVertexValid(pt):
            continue
        vertex = np.vstack((vertex, pt))
    return vertex


def get_edges(vertex):
    size = vertex.shape[0]
    edges = []
    for i in range(size):
        edges.append([])
    for i in range(size):
        for j in range(i + 1, size):
            p1 = vertex[i, :]
            p2 = vertex[j, :]
            if IsLineValid(p1, p2):
                edges[i].append(j)
                edges[j].append(i)
    return edges


class PriorityQueue(object):

    def __init__(self, node):
        self._queue = sortedcontainers.SortedList([node])

    def push(self, node):
        self._queue.add(node)

    def pop(self):
        return self._queue.pop(index=0)

    def empty(self):
        return len(self._queue) == 0

    def compare_and_replace(self, i, node):
        if node < self._queue[i]:
            self._queue.pop(index=i)
            self._queue.add(node)

    def find(self, node):
        try:
            loc = self._queue.index(node)
            return loc
        except ValueError:
            return None


class Node(object):
    def __init__(self, vertex_order=0, g=0, h=0, parent=None):
        self.vertex_order = vertex_order
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.vertex_order == other.vertex_order


def resample(path):
    for i in range(len(path) - 1):
        dis = distance(path[i], path[i + 1])
        if i == 0:
            obj_x = np.linspace(path[i][0], path[i + 1][0], round(dis))
            obj_x = np.delete(obj_x, -1)
            obj_y = np.linspace(path[i][1], path[i + 1][1], round(dis))
            obj_y = np.delete(obj_y, -1)
        elif i == len(path) - 2:
            obj_x = np.hstack((obj_x, np.linspace(path[i][0], path[i + 1][0], round(dis))))
            obj_y = np.hstack((obj_y, np.linspace(path[i][1], path[i + 1][1], round(dis))))
        else:
            obj_x = np.hstack((obj_x, np.linspace(path[i][0], path[i + 1][0], round(dis))))
            obj_x = np.delete(obj_x, -1)
            obj_y = np.hstack((obj_y, np.linspace(path[i][1], path[i + 1][1], round(dis))))
            obj_y = np.delete(obj_y, -1)
    path = np.vstack((obj_x, obj_y)).T
    return path


def PRM():
    vertex = getvertex(100)
    print('get vertex over')
    edges = get_edges(vertex)
    print('get edges over')
    node = Node(h=distance(vertex[0], vertex[1]))
    completeness_flag = 0
    path_length = 0
    iter_num = 0
    max_iter = 500
    open_list = PriorityQueue(node)
    closed = set()
    path = []
    prev = None
    while (not open_list.empty()) and (not completeness_flag) and (iter_num < max_iter):
        cur_node = open_list.pop()
        closed.add(cur_node.vertex_order)
        for edge in range(len(edges[cur_node.vertex_order])):
            newVertex = edges[cur_node.vertex_order][edge]
            child_g = cur_node.g + distance(vertex[cur_node.vertex_order], vertex[newVertex])
            child_h = distance(vertex[1], vertex[newVertex])
            child_node = Node(vertex_order=newVertex, g=child_g, h=child_h, parent=cur_node)
            if newVertex not in closed and not open_list.find(child_node):
                prev = child_node
                if child_node.vertex_order == 1:
                    path_length = child_node.f
                    completeness_flag = 1
                    break
                open_list.push(child_node)
        iter_num += 1

    while prev is not None:
        curp = vertex[prev.vertex_order]
        path.append(curp)
        prev = prev.parent
    path = list(reversed(path))
    if completeness_flag:
        path = resample(path)
        print('succeed')
    else:
        print('failed')
        return
    x = path[:, 0]
    y = path[:, 1]
    theta = []
    for i in range(len(path) - 1):
        theta.append(np.arctan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0]))
    theta.append(0)
    theta = np.array(theta)
    return x, y, theta, path_length, completeness_flag


if __name__ == "__main__":
    globalvar.obstacles_ = GenerateStaticObstacles_unstructured()
    [x, y, theta, path_length, completeness_flag] = PRM()
    transmethod_flag = 1  # choose different path to trajectory method
    trajectory = TransformPathToTrajectory(x, y, theta, path_length, transmethod_flag)
    VisualizeStaticResults(trajectory)
    VisualizeDynamicResults(trajectory)



