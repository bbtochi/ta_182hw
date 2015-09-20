# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

from util import *

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, `stepCost`), where 'successor' is a
        successor to the p_lastrent state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    "*** YOUR CODE HERE ***"
    frontier, explored, s = Stack(), [], problem.getStartState()
    frontier.push([(s,'',0)])
    while not frontier.isEmpty():
        path = frontier.pop()
        p_last = path[-1][0]
        if problem.isGoalState(p_last):
            return [state[1] for state in path[1:]]
        explored.append(p_last)
        for neighbor in problem.getSuccessors(p_last):
            if neighbor[0] not in explored:
                frontier.push(path+[neighbor])

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """

    "*** YOUR CODE HERE ***"
    frontier, explored, s = Queue(), [], problem.getStartState()
    frontier.push([(s,'',0)])
    while not frontier.isEmpty():
        path = frontier.pop()
        p_last = path[-1][0]
        if problem.isGoalState(p_last):
            return [state[1] for state in path[1:]]
        explored.append(p_last)
        for neighbor in problem.getSuccessors(p_last):
            if neighbor[0] not in explored:
                frontier.push(path+[neighbor])

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    frontier, s = PriorityQueue(), problem.getStartState()
    path, explored, cost = [(s,'',0)], [], {}
    priority = lambda pth: reduce(lambda x,y: x+y, map(lambda state: state[2],pth))
    frontier.push(path, priority(path))
    # path_cost[s] = priority(path)
    while not frontier.isEmpty():
        path = frontier.pop()
        p_last = path[-1][0]
        if problem.isGoalState(p_last):
            return [state[1] for state in path[1:]]
        explored.append(p_last)
        for neighbor in problem.getSuccessors(p_last):
            next_path = path+[neighbor]
            if neighbor[0] not in explored:
                frontier.push(next_path,priority(next_path))
            #     path_cost[neighbor] = priority(next_path)
            # elif priority(next_path) < path_cost[neighbor]:
            #     print 'hi'
            #     frontier.push(next_path,priority(next_path))
            #     path_cost[neighbor] = priority(next_path)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the p_lastrent state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    frontier, s = PriorityQueue(), problem.getStartState()
    path, explored, cost = [(s,'',0)], [], {}
    priority = lambda pth: reduce(lambda x,y: x+y, map(lambda state: state[2],pth))+heuristic(pth[-1][0],problem)
    frontier.push(path, priority(path))
    while not frontier.isEmpty():
        path = frontier.pop()
        p_last = path[-1][0]
        if problem.isGoalState(p_last):
            return [state[1] for state in path[1:]]
        explored.append(p_last)
        for neighbor in problem.getSuccessors(p_last):
            next_path = path+[neighbor]
            if neighbor[0] not in explored:
                frontier.push(next_path,priority(next_path))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
