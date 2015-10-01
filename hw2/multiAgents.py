"""
    PROBLEM SET COMPLETED BY TOCHI ONYENOKWE AND ALEX SAICH
"""
# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from sys import maxint

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos, curPos = successorGameState.getPacmanPosition(), currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates, curGhostStates = successorGameState.getGhostStates(), currentGameState.getGhostStates()
        newUnscared, curUnscared = [g for g in newGhostStates if g.scaredTimer < 2.], [g for g in curGhostStates if g.scaredTimer < 2.]


        "*** YOUR CODE HERE ***"
        """
        Thoughts:
        * Moves that get you very close to ghosts should have lower scores
        * Except when ghosts are scared
        * Moves that get you closer to food should have higher scores
        * Might also want to consider direction of pacman and nearest ghost
        """
        # initialize score to be current game score
        score = successorGameState.getScore()

        # get distance to closest food in the new gamestate
        newFoodDist = 1.         # if new pos has food on it this will remain 1
        # if new position doesn't have food on it
        if not (currentGameState.hasFood(*newPos)):
            newFoodDist = min( map(lambda x: manhattanDistance(newPos, x), newFood.asList()) )
        score += 1./newFoodDist

        # initialize distance to closest unfrightened ghost in current and new gamestate
        curGhostDist = newGhostDist = 1000
        # if there are any unscared ghosts in current or new gamestate reset distance to closest one
        if len(curUnscared) != 0:
            curGhostDist = min( map(lambda x: manhattanDistance(curPos, x.getPosition()), curUnscared) )
        if len(newUnscared) != 0:
            newGhostDist = min( map(lambda x: manhattanDistance(newPos, x.getPosition()), newUnscared) )

        # if pacman is in danger zone (i.e. too close to ghost)
        if curGhostDist < 3.:
            # if next position gets you further away form nearest unscared ghost
            if newGhostDist > curGhostDist:
                score+=60.
            else:
                score-=60.
        # else if next move gets you into danger zone
        elif newGhostDist < 3.:
            score-=40.

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # number of agents at current gameState
        numAgents = gameState.getNumAgents()
        def minimax(agent,state,curDepth):
            # if at the limit of specified depth
            if curDepth == 0:
                # return score of current state
                return ('',self.evaluationFunction(state))

            # set the extreme value based on what agent has a turn
            val = ('',float(-1*maxint)) if agent == 0 else ('',float(maxint))

            actions = state.getLegalActions(agent)
            # if there are no possible actions return score of current state
            if len(actions) == 0:
                val = ('', self.evaluationFunction(state))
            else:
                # if we are at the last agent update the depth for the next turn
                if agent == numAgents-1: curDepth-=1
                for a in actions:
                    successor, nextAgent = state.generateSuccessor(agent,a), (agent+1)%numAgents
                    util = minimax(nextAgent,successor,curDepth)[1]
                    # based on what agent has a turn, update the current best 'utility'
                    if agent == 0 and val[1] < util: val = (a,util)
                    elif agent != 0 and val[1] > util: val = (a,util)
            return val
        return minimax(0,gameState,self.depth)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        Alpha, Beta, numAgents = float(-1*maxint), float(maxint), gameState.getNumAgents()
        def AlphaBetaMax(agent,state,curDepth,alpha,beta):
            if curDepth == 0: return ('',self.evaluationFunction(state))

            val = ('',float(-1*maxint)) if agent == 0 else ('',float(maxint))
            actions = state.getLegalActions(agent)
            if len(actions) == 0:
                val = ('', self.evaluationFunction(state))
            else:
                if agent == numAgents-1: curDepth-=1
                for a in actions:
                    successor, nextAgent = state.generateSuccessor(agent,a), (agent+1)%numAgents
                    util = AlphaBetaMax(nextAgent,successor,curDepth,alpha,beta)[1]
                    # if pacman turn
                    if agent == 0:
                        if val[1] < util: val = (a,util)
                        # check if we can prune
                        if val[1] > beta: return (a, val[1])
                        alpha = max(alpha,val[1])
                    else:
                        if val[1] > util: val = (a,util)
                        if val[1] < alpha: return (a, val[1])
                        beta = min(beta,val[1])
            return val
        return AlphaBetaMax(0,gameState,self.depth,Alpha,Beta)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        def expectimax(agent,state,curDepth):
            if curDepth == 0: return ('',self.evaluationFunction(state))

            val = ('',float(-1*maxint)) if agent == 0 else ('',0.)
            actions = state.getLegalActions(agent)
            if len(actions) == 0:
                val = ('', self.evaluationFunction(state))
            else:
                if agent == numAgents-1: curDepth-=1
                exp_val = val[1]
                for a in actions:
                    successor, nextAgent = state.generateSuccessor(agent,a), (agent+1)%numAgents
                    util = expectimax(nextAgent,successor,curDepth)[1]
                    if agent == 0 and val[1] < util: val = (a,util)
                    elif agent != 0:
                        # update average utility of actions seen
                        exp_val += (float(util) * (1./float(len(actions)))); val = (a,exp_val)
            return val
        return expectimax(0,gameState,self.depth)[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    """
        THOUGHTS
        * If a state is closer to more food then it should have a higher score
        * If a state is too close to a ghost it should have a lower score
        * Maybe take the average over all the scores of each action
    """

    scores, actions = [], currentGameState.getLegalActions()

    # if there are no legal actions return score of current state
    if len(actions) == 0:
        return currentGameState.getScore()

    for action in actions:
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos, curPos = successorGameState.getPacmanPosition(), currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates, curGhostStates = successorGameState.getGhostStates(), currentGameState.getGhostStates()
        newUnscared, curUnscared = [g for g in newGhostStates if g.scaredTimer < 2.], [g for g in curGhostStates if g.scaredTimer < 2.]

        # initialize score
        score = successorGameState.getScore()

        # get reciprocal distance to closest food in new and current gamestates
        newFoodDist = 1.
        # if new position doesn't have food on it
        if not (currentGameState.hasFood(*newPos)):
            newFoodDist = min( map(lambda x: manhattanDistance(newPos, x), newFood.asList()) )
        score += 1./newFoodDist

        # set distance to closest unfrightened ghost in current and new gamestate as 0
        curGhostDist = newGhostDist = 1000
        # if there are any unscared ghosts in current or new gamestate
        if len(curUnscared) != 0:
            curGhostDist = min( map(lambda x: manhattanDistance(curPos, x.getPosition()), curUnscared) )
        if len(newUnscared) != 0:
            newGhostDist = min( map(lambda x: manhattanDistance(newPos, x.getPosition()), newUnscared) )

        # if pacman is in danger zone (too close to ghost)
        if curGhostDist < 3.:
            # if next position gets you further away form nearest unscared ghost
            if newGhostDist > curGhostDist:
                score+=60.
            else:
                score-=60.
        # else if next move gets you in danger zone
        elif newGhostDist < 3.:
            score-=40.
        scores.append(score)

    # take the average over scores for all possible actions
    return reduce(lambda x,y: x+y,scores)/float(len(scores))

# Abbreviation
better = betterEvaluationFunction
