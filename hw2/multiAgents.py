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
        newFood, curFood = successorGameState.getFood(), currentGameState.getFood()
        newGhostStates, curGhostStates = successorGameState.getGhostStates(), currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newUnscared, curUnscared = [g for g in newGhostStates if g.scaredTimer < 2.], [g for g in curGhostStates if g.scaredTimer < 2.]

        # print
        # print "Pacman Direction",(successorGameState.getPacmanState().getDirection())
        # print "Ghost Direction", (newGhostStates[0].getDirection())
        # print "New Scared Times:", newScaredTimes
        # print "New Food", len((newFood).asList())
        # print "New Pos", newPos


        "*** YOUR CODE HERE ***"
        """
        Thoughts:
        * Moves that get you very close to ghosts should have lower scores
        * Except when ghosts are scared
        * Moves that get you closer to food should have higher scores
        * Might also want to consider direction of pacman and nearest ghost
        """
        # initialize score
        score = successorGameState.getScore()

        # get reciprocal distance to closest food in new and current gamestates
        newFoodDist = 1
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
        # print "DEPTH:", self.depth
        # print "NUMBER OF AGENTS", gameState.getNumAgents()
        # print "LEGAL ACTIONS AGENT 0", gameState.getLegalActions(0)
        numAgents = gameState.getNumAgents()

        def minimax(agent,state,curDepth):
            if curDepth == 0:
                return ('Stop',self.evaluationFunction(state))
            if agent == 0:
                actions = state.getLegalActions(0)
                val = 'Stop'
                if len(actions) == 0:
                    val = ('Stop', self.evaluationFunction(state))

                for a in actions:
                    successor = state.generateSuccessor(agent,a)
                    util = minimax(1,successor,curDepth)[1]
                    if val == 'Stop':
                        val = (a,util)
                    else:
                        if float(val[1]) < float(util):
                            val = (a,util)

                return val
            else:
                actions = state.getLegalActions(agent)
                val = 'Stop'
                if agent == numAgents-1: curDepth-=1

                if len(actions) == 0:
                    val = ('Stop', self.evaluationFunction(state))

                for a in actions:
                    successor = state.generateSuccessor(agent,a)
                    util = minimax((agent+1)%numAgents,successor,curDepth)[1]
                    if val == 'Stop':
                        val = (a,util)
                    else:
                        if float(val[1]) > float(util):
                            val = (a,util)
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
        alpha = -9000000.
        beta = 9000000.
        numAgents = gameState.getNumAgents()


        def ab(alpha, beta, agent,state,curDepth):
            if curDepth == 0:
                return ('Stop',self.evaluationFunction(state))
            
            if agent == 0:
                actions = state.getLegalActions(0)
                val = 'Stop'
                if len(actions) == 0:
                    val = ('Stop', self.evaluationFunction(state))

                for a in actions:
                    successor = state.generateSuccessor(agent,a)
                    util = ab(alpha, beta, 1,successor,curDepth)[1]
                    
                    if val == 'Stop':
                        val = (a,util)
                    else:
                        if float(val[1]) < float(util):
                            val = (a,util)
                    
                    alpha = max(float(alpha), float(val[1]))
                    if val[1] >= beta: return (a, val[1])

                return val
            else:
                actions = state.getLegalActions(agent)
                val = 'Stop'
                if agent == numAgents-1: curDepth-=1

                if len(actions) == 0:
                    val = ('Stop', self.evaluationFunction(state))

                for a in actions:
                    successor = state.generateSuccessor(agent,a)
                    util = ab(alpha, beta, (agent+1)%numAgents,successor,curDepth)[1]
                    
                    if val == 'Stop':
                        val = (a,util)
                    else:
                        if float(val[1]) > float(util):
                            val = (a,util)
                            
                    beta = min(float(beta), float(val[1]))
                    if val[1] <= alpha: return (a, val[1])
                
                return val

        return ab(alpha, beta, 0, gameState,self.depth)[0]

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
