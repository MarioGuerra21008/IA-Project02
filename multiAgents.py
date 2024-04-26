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
import random, util, math

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        ghostDistance = self.distanceToTheNearestGhost(childGameState)
        foodDistance = self.distanceToTheNearestFood(childGameState, currentGameState)
        return ghostDistance / (foodDistance + 1)
    
    def distanceToTheNearestGhost(self, gameState):
        pacmanPosition = gameState.getPacmanPosition()
        ghostsPosition = gameState.getGhostPositions()
        distance = [math.dist(pacmanPosition, ghostPosition) for ghostPosition in ghostsPosition]
        return min(distance)

    def distanceToTheNearestFood(self, nextState, currentState):
        pacmanPosition = nextState.getPacmanPosition()
        foodPositions = currentState.getFood().asList()
        distance = [math.dist(pacmanPosition, foodPosition) for foodPosition in foodPositions]
        return min(distance)

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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        action, _ = self.minimaxValue(gameState, 0, 0)
        return action
    
    def minimaxValue(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth >= self.depth * gameState.getNumAgents():
            return 'Stop', self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.bestValues(gameState, 0, depth, function=max)
        else:
            return self.bestValues(gameState, agentIndex, depth, function=min)
        
    def bestValues(self, gameState, agentIndex, depth, function):
        nextActions = gameState.getLegalActions(agentIndex)
        nextStates = [gameState.getNextState(agentIndex, action) for action in nextActions]
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        values = [self.minimaxValue(nextState, nextAgentIndex, depth+1)[1] for nextState in nextStates]
        bestValue = function(values)
        bestAction = nextActions[values.index(bestValue)]

        return bestAction, bestValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def AB(gameState,agent,depth,a,b):
            result = []

            # Terminate state #
            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            # Reached max depth #
            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            # All ghosts have finised one round: increase depth #
            if agent == gameState.getNumAgents() - 1:
                depth += 1

            # Calculate nextAgent #

            # Last ghost: nextAgent = pacman #
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            # Availiable ghosts. Pick next ghost #
            else:
                nextAgent = agent + 1

            # For every successor find minmax value #
            for action in gameState.getLegalActions(agent):
                if not result: # First move
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

                    # Fix result #
                    result.append(nextValue[0])
                    result.append(action)

                    # Fix initial a,b (for the first node) #
                    if agent == self.index:
                        a = max(result[0],a)
                    else:
                        b = min(result[0],b)
                else:
                    # Check if minMax value is better than the previous one #
                    # Chech if we can overpass some nodes                   #

                    # There is no need to search next nodes                 #
                    # AB Prunning is true                                   #
                    if result[0] > b and agent == self.index:
                        return result

                    if result[0] < a and agent != self.index:
                        return result

                    previousValue = result[0] # Keep previous value
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

                    # Max agent: Pacman #
                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            # a may change #
                            a = max(result[0],a)

                    # Min agent: Ghost #
                    else:
                        if nextValue[0] < previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            # b may change #
                            b = min(result[0],b)
            return result

        # Call AB with initial depth = 0 and -inf and inf(a,b) values      #
        # Get an action                                                    #
        # Pacman plays first -> self.index                                #
        return AB(gameState,self.index,0,-float("inf"),float("inf"))[1]



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
        return self.expectimaxValue(gameState, 0, 0)[0]

    def expectimaxValue(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth >= self.depth * gameState.getNumAgents():
            return 'Stop', self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.expectation(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        legalActions = gameState.getLegalActions(agentIndex)
        if not legalActions:
            return 'Stop', self.evaluationFunction(gameState)
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        values = [(action, self.expectimaxValue(gameState.generateSuccessor(agentIndex, action), nextAgentIndex, depth+1)[1]) for action in legalActions]
        return max(values, key=lambda x: x[1])

    def expectation(self, gameState, agentIndex, depth):
        legalActions = gameState.getLegalActions(agentIndex)
        if not legalActions:
            return 'Stop', self.evaluationFunction(gameState)
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        values = [self.expectimaxValue(gameState.generateSuccessor(agentIndex, action), nextAgentIndex, depth+1)[1] for action in legalActions]
        return 'Stop', sum(values) / len(values)









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
