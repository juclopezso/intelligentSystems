# bustersAgents.py
# ----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False): pass
    def update(self, state):    pass
    def pause(self):    pass
    def draw(self, state):    pass
    def updateDistributions(self, dist):    pass
    def finish(self):    pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0: allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
	"An agent that tracks and displays its beliefs about ghost positions."

	def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
		inferenceType = util.lookup(inference, globals())
		self.inferenceModules = [inferenceType(a) for a in ghostAgents]
		self.observeEnable = observeEnable
		self.elapseTimeEnable = elapseTimeEnable

	def registerInitialState(self, gameState):
		"Initializes beliefs and inference modules"
		import __main__
		self.display = __main__._display
		for inference in self.inferenceModules: inference.initialize(gameState)
		self.beliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
		self.wallBeliefs = self.beliefs[0]
		self.firstMove = True

	def observationFunction(self, gameState):
		"Removes the ghost states from the gameState"
		agents = gameState.data.agentStates
		gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
		return gameState

	def getAction(self, gameState):

		"Updates beliefs, then chooses an action based on updated beliefs."
		for index, inf in enumerate(self.inferenceModules):
			if not self.firstMove and self.elapseTimeEnable:
				inf.elapseTime(gameState)
			self.firstMove = False
			if self.observeEnable:
				inf.observeState(gameState)
			self.beliefs[index] = inf.getBeliefDistribution()
			self.wallBeliefs = self.beliefs[0]
		self.display.updateDistributions([self.wallBeliefs])
		
		return self.chooseAction(gameState)

	

	def chooseAction(self, gameState):
		"By default, a BustersAgent just stops.  This should be overridden."
		return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions

class GreedyBustersAgent(BustersAgent):
	"An agent that charges the closest ghost."
	def registerInitialState(self, gameState):
		BustersAgent.registerInitialState(self, gameState)


	def getGhostPosition(self):
		return [ x[0].getPosition() for x in self.display.agentImages if not x[0].isPacman and x[0].getPosition()[1] != 1 ][0]


	def chooseAction(self, gameState):
		"""
		1. Given de belief distribution choose the position with the maximum probability.
		If there is a draw choose the position with the maximum position according to lexicographic order of the
		location.

		Lexicographic order is defined as follows, a pair (X2,Y2) is greater lexicographically than (X1,Y1) if and
		only if: x1 < x2 OR ( x2 >= x1 AND y1 < y2)

		2. From this position, choose the action that brings you closer to the ghost,
		i.e. the position with the minimum distance, in terms of the number of steps in the maze,
		to the position of the ghost.

		In oder to access the beliefs you must use self.wallBeliefs, this object is passed from the inference
		module you previously defined.

		You can access to the map via gameState.getWalls() or gameState.hasWall(x,y). Remember you must NOT
		make any assumptions about the pacman position beside what you can infer from the beliefs.
		"""
		#TODO: Put your code here
		util.raiseNotDefined()
	
		


