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

import numpy as np

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"

        closeGhost = closestGhostDist(newGhostStates, newPos)
        ghostCost = -50 / (np.e ** closeGhost)
        if closeGhost <= 1:
            ghostCost += -1500
        foodCost = - closest_food_dist(newFood.asList(), newPos)
        foodCount = -30 * len(newFood.asList())


        return ghostCost + foodCost


def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclid_dist(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def closest_food_dist(food, pos):
    if len(food) > 0:
        food.sort(key=lambda x: manhattan_dist(pos, x))
        return manhattan_dist(pos, food[0])
    else:
        return 0


def closestGhostDist(ghosts, pos):
    if len(ghosts) > 0:
        minDist = float('inf')
        for ghost in ghosts:
            dist = manhattan_dist(pos, ghost.getPosition())
            if dist < minDist:
                minDist = dist
        return minDist
    else:
        return 0


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        depth = 0
        maxVal = -1145141919
        bestAction = None
        ghostNum = gameState.getNumAgents() - 1
        agentIndex = 0

        successor = {x: x+1 for x in range(ghostNum)}
        successor[ghostNum] = 0
        nextAgent = successor[agentIndex]

        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            actionVal = self.minNode(depth, nextAgent, nextState, successor)
            if actionVal > maxVal:
                maxVal = actionVal
                bestAction = action
        return bestAction


    def minNode(self, depth, agentIndex, gameState, successor):
        minVal = 1145141919
        nextAgent = successor[agentIndex]

        if gameState.getLegalActions(agentIndex) == []:
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            if nextAgent == 0:
                minVal = min(minVal, self.maxNode(depth, nextAgent, nextState, successor))
            else:
                minVal = min(minVal, self.minNode(depth, nextAgent, nextState, successor))

        return minVal


    def maxNode(self, depth, agentIndex, gameState, successor):
        maxVal = -1145141919
        nextAgent = successor[agentIndex]
        depth += 1

        if gameState.getLegalActions(agentIndex) == []:
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            maxVal = max(maxVal, self.minNode(depth, nextAgent, nextState, successor))

        return maxVal


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        depth = 0
        maxVal = -1145141919
        bestAction = None
        ghostNum = gameState.getNumAgents() - 1
        agentIndex = 0
        alpha = -1145141919
        beta = 1145141919

        successor = {x: x+1 for x in range(ghostNum)}
        successor[ghostNum] = 0
        nextAgent = successor[agentIndex]

        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            actionVal = self.minNode(depth, nextAgent, nextState, successor, alpha, beta)
            if actionVal > maxVal:
                maxVal = actionVal
                alpha = actionVal
                bestAction = action
        return bestAction


    def minNode(self, depth, agentIndex, gameState, successor, alpha, beta):
        minVal = 1145141919
        nextAgent = successor[agentIndex]

        if gameState.getLegalActions(agentIndex) == []:
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            if nextAgent == 0:
                minVal = min(minVal, self.maxNode(depth, nextAgent, nextState, successor, alpha, beta))
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)
            else:
                minVal = min(minVal, self.minNode(depth, nextAgent, nextState, successor, alpha, beta))
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)

        return minVal


    def maxNode(self, depth, agentIndex, gameState, successor, alpha, beta):
        maxVal = -1145141919
        nextAgent = successor[agentIndex]
        depth += 1

        if gameState.getLegalActions(agentIndex) == []:
            return self.evaluationFunction(gameState)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            maxVal = max(maxVal, self.minNode(depth, nextAgent, nextState, successor, alpha, beta))
            if maxVal > beta:
                return maxVal
            alpha = max(alpha, maxVal)

        return maxVal


class ExpectimaxAgent(MinimaxAgent):
    """
      Your expectimax agent (question 4)
    """

    def minNode(self, depth, agentIndex, gameState, successor):
        expectVal = []
        nextAgent = successor[agentIndex]

        if gameState.getLegalActions(agentIndex) == []:
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            if nextAgent == 0:
                expectVal.append(self.maxNode(depth, nextAgent, nextState, successor))
            else:
                expectVal.append(self.minNode(depth, nextAgent, nextState, successor))
        return np.mean(expectVal)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodCount= len(newFood.asList())

    closeGhost = closestGhostDist(newGhostStates, newPos)
    ghostCost = -10/(np.e ** (10 * closeGhost))
    if closeGhost <= 0:
        ghostCost = -200000
        if newScaredTimes[0] > 1:
            ghostCost = 200000
    foodCost =  - 50 * closest_food_dist(newFood.asList(), newPos)
    foodCountCost = -50000 * foodCount
    scaredTimeCost = 100 * newScaredTimes[0]

    return ghostCost + foodCost + foodCountCost


# Abbreviation
better = betterEvaluationFunction
