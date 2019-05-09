import sys
import bisect
import random
# from pip._vendor.msgpack.fallback import xrange
infinity = float('inf')  # sistemski definirana vrednost za beskonecnost


class Queue:

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)


def Stack():
    return []


class FIFOQueue(Queue):

    def __init__(self):
        self.A = []
        self.start = 0

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A) - self.start

    def extend(self, items):
        self.A.extend(items)

    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A) / 2:
            self.A = self.A[self.start:]
            self.start = 0
        return e

    def __contains__(self, item):
        return item in self.A[self.start:]


class PriorityQueue(Queue):

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)



class Problem:

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        raise NotImplementedError

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self):
        raise NotImplementedError


class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        return [node.action for node in self.path()[1:]]

    def solve(self):
        return [node.state for node in self.path()[0:]]

    def path(self):
        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        return list(reversed(result))


    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


def graph_search(problem, fringe):
    closed = {}
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node
        if node.state not in closed:
            closed[node.state] = True
            fringe.extend(node.expand(problem))
    return None


def memoize(fn, slot=None):
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}
    return memoized_fn


def best_first_graph_search(problem, f):

    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def astar_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


# Graphs and Graph Problems
import math


def distance(a, b):
    return math.hypot((a[0] - b[0]), (a[1] - b[1]))


class Graph:

    def __init__(self, dict=None, directed=True):
        self.dict = dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        for a in list(self.dict.keys()):
            for (b, dist) in self.dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        self.dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        links = self.dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        return list(self.dict.keys())


def UndirectedGraph(dict=None):
    return Graph(dict=dict, directed=False)



class GraphProblem(Problem):

    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or infinity)

    def h(self, node):
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity

Pocetok = input()
Stanica1 = input()
Stanica2 = input()
Kraj = input()

ABdistance=distance((2,1),(2,4))
BIdistance=distance((2,4),(8,5))
BCdistance=distance((2,4),(2,10))
HIdistance=distance((8,1),(8,5))
IJdistance=distance((8,5),(8,8))
FCdistance=distance((5,9),(2,10))
GCdistance=distance((4,11),(2,10))
CDdistance=distance((2,10),(2,15))
FGdistance=distance((5,9),(4,11))
FJdistance=distance((5,9),(8,8))
KGdistance=distance((8,13),(4,11))
LKdistance=distance((8,15),(8,13))
LMdistance=distance((8,15),(8,19))
DEdistance=distance((2,15),(2,19))
DLdistance=distance((2,15),(8,15))


graph = UndirectedGraph({
    "B": {"A": ABdistance, "I": BIdistance, "C": BCdistance},
    "I": {"H": HIdistance, "J": IJdistance},
    "C": {"F": FCdistance, "G": GCdistance, "D": CDdistance},
    "F": {"G": FGdistance, "J": FJdistance},
    "K": {"G": KGdistance, "L": LKdistance},
    "D": {"E": DEdistance, "L": DLdistance},
    "M": {"L": LMdistance}
})


graph.locations = dict(
A = (2,1) , B = (2,4) , C = (2,10) ,
D = (2,15) , E = (2,19) , F = (5,9) ,
G=(4,11) , H = (8,1) , I = (8,5),
J = (8,8) , K = (8,13) , L = (8,15),
M = (8,19))

graph1Problem=GraphProblem(Pocetok,Stanica1,graph)
rez1=astar_search(graph1Problem)

graph2Problem=GraphProblem(Stanica1,Kraj,graph)
rez2=astar_search(graph2Problem)

grap3Problem=GraphProblem(Pocetok,Stanica2,graph)
rez3=astar_search(grap3Problem)

graph4Problem=GraphProblem(Stanica2,Kraj,graph)
rez4=astar_search(graph4Problem)

rez=None

if rez1.path_cost+rez2.path_cost>rez3.path_cost+rez4.path_cost:
    rez3=rez3.solve()
    rez4=rez4.solve()
    rez=rez3+rez4[1:len(rez4)]
else:
    rez1=rez1.solve()
    rez2=rez2.solve()
    rez=rez1+rez2[1:len(rez2)]


print(rez)