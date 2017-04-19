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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #Create list of all capsules
        allCapsules = successorGameState.getCapsules()

        val = successorGameState.getScore()


        distGhost = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if (distGhost > 0):
        	val -= 10 / distGhost

        #List of distance to capsule	
        capsuleDistances = [manhattanDistance(newPos, i) for i in allCapsules]
        

        #List of distances to food
        foodDistances = [manhattanDistance(newPos, x) for x in newFood.asList()]

        if len(foodDistances) > 0:
        	val += 10 / min(foodDistances)

        if len(capsuleDistances) > 0:
        	val -= 5 / min(capsuleDistances)

        return val


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

	def getAction(self, gameState):


		def max_val(state, currDepth):
			#Max is called when we look one depth more, so add 1 to restart the min/max process
			currDepth = currDepth + 1
			#If terminal, return the eval function; currDepth == self.depth is also base case that stops recursion
			if (state.isWin() or state.isLose() or currDepth == self.depth):
				return self.evaluationFunction(state)
			v = float('-Inf')

			"""For every action in Pacman's legal actions, make the current v equal to the maximum value 
			between -infinity and the minval of their successors (ghosts, so index = 1, at that current Depth"""
			for action in state.getLegalActions(0):
				v = max(v, min_val(state.generateSuccessor(0, action), currDepth, 1))
			return v

		def min_val(state, currDepth, ghostIndex):
			#If terminal, return the eval function
			if (state.isWin() or state.isLose()):
				return self.evaluationFunction(state)
			v = float('Inf')

			"""For every action in the legal actions of the current ghost, if the current ghost is the last 
			ghost depth layer, return the min of the max of the successors; if it is not then return the
			min of the min of the successors until you get to the last ghost"""
			for action in state.getLegalActions(ghostIndex):
				#Base case, will equal when ghostIndex is last depth level
				if (ghostIndex == state.getNumAgents() - 1):
					v = min(v, max_val(state.generateSuccessor(ghostIndex, action), currDepth))
				else:
					v = min(v, min_val(state.generateSuccessor(ghostIndex, action), currDepth, ghostIndex + 1))
			return v

		#All legal pacman moves
		legalActions = gameState.getLegalActions(0)

		#Set initial v and return action to -infinity and no move
		maxV = float('-Inf')
		maxAction = ''

		"""Loop through every action, starting off with current depth = 0 and current max value = minimum of the
		successors of pacman at that action, the depth, and ghost 1; if the current max is greater than the overall V,
		reassign maxV and maxAction; loop through all actions then return the maxAction"""
		for action in legalActions:
			currDepth = 0
			currMax = min_val(gameState.generateSuccessor(0, action), currDepth, 1)
			if currMax > maxV:
				maxV = currMax
				maxAction = action
		return maxAction



class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		def max_val(state, alpha, beta, currDepth):
			currDepth = currDepth + 1
			if (state.isWin() or state.isLose() or currDepth == self.depth):
				return self.evaluationFunction(state)
			v = float('-Inf')

			for action in state.getLegalActions(0):
				v = max(v, min_val(state.generateSuccessor(0, action), alpha, beta, currDepth, 1))
				if v > beta:
					return v
				alpha = max(v, alpha)
			return v


		def min_val(state, alpha, beta, currDepth, ghostIndex):
			if (state.isWin() or state.isLose()):
				return self.evaluationFunction(state)
			v = float('Inf')

			for action in state.getLegalActions(ghostIndex):
				if (ghostIndex == state.getNumAgents() - 1):
					v = min(v, max_val(state.generateSuccessor(ghostIndex, action), alpha, beta, currDepth))
				else:
					v = min(v, min_val(state.generateSuccessor(ghostIndex, action), alpha, beta, currDepth, ghostIndex + 1))
				if v < alpha:
					return v
				beta = min(v, beta)
			return v

		legalActions = gameState.getLegalActions(0)
		maxV = float('-Inf')
		alpha = float('-Inf')
		beta = float('Inf')
		maxAction = legalActions[0]

		for action in legalActions:
			#currDepth = 0
			currMax = min_val(gameState.generateSuccessor(0, action), alpha, beta, 0, 1)
			if (currMax > maxV):
				maxV = currMax
				maxAction = action
			if (currMax > beta):
				return maxAction
			alpha = max(alpha, currMax)
		return maxAction




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
		def max_val(state, currDepth):
			currDepth = currDepth + 1
			if (state.isWin() or state.isLose() or currDepth == self.depth):
				return self.evaluationFunction(state)
			v = float('-Inf')

			for action in state.getLegalActions(0):
				v = max(v, exp_val(state.generateSuccessor(0, action), currDepth, 1)) 
			return v


		def exp_val(state, currDepth, ghostIndex):
			#If terminal, return the eval function
			if (state.isWin() or state.isLose()):
				return self.evaluationFunction(state)
			v = 0.0

			for action in state.getLegalActions(ghostIndex):
				if (ghostIndex == state.getNumAgents() - 1):
					v += float(max_val(state.generateSuccessor(ghostIndex, action), currDepth) / len(state.getLegalActions(ghostIndex)))
				else:
					v += float(exp_val(state.generateSuccessor(ghostIndex, action), currDepth, ghostIndex + 1) / len(state.getLegalActions(ghostIndex)))
			return v 

		legalActions = gameState.getLegalActions(0)
		maxV = float('-inf')
		maxAction = ''

		for action in legalActions:
			currDepth = 0
			currMax = exp_val(gameState.generateSuccessor(0, action), currDepth, 1)
			if currMax > maxV:
				maxV = currMax
				maxAction = action
		return maxAction



def betterEvaluationFunction(currentGameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (question 5).

	DESCRIPTION: Used current game state and gave the distances to ghosts a negative weight, and distances to food a positive weight
	"""
	currPos = currentGameState.getPacmanPosition()
	currFood = currentGameState.getFood()
	ghostStates = currentGameState.getGhostStates()
	scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]



	val = currentGameState.getScore()


	distGhost = manhattanDistance(currPos, ghostStates[0].getPosition())
	if (distGhost > 0):
		val -= 10 / distGhost

	#List of distances to food
	foodDistances = [manhattanDistance(currPos, x) for x in currFood.asList()]



	if len(foodDistances) > 0:
		val += 10 / min(foodDistances)


	return val


# Abbreviation
better = betterEvaluationFunction

