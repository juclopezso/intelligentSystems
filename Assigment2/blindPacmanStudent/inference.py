# inference.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import itertools
import util
import random
import busters
import game
import random
from game import Directions
from game import Actions

def neighbors(position,gameState):

        adj = 0
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            successorPosition = game.Actions.getSuccessor(position,action)
            if not gameState.hasWall(int(successorPosition[0]),int(successorPosition[1])):
                adj += 1

        return adj > 0


class InferenceModule:
    """
    An inference module tracks a belief distribution over pacman location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = [] # most recent observation position

    
    def getPacmanSuccesorDistribution(self,gameState,position):

        """
        Returns a distribution over successor positions of the pacman from the given gameState.
        No assumptions are made about the way Pacman is moving.
        """

        dist = util.Counter()

        actions = []

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            successorPosition = game.Actions.getSuccessor(position,action)
            x ,y = map(int,successorPosition)
            if gameState.hasWall(x,y) == False:
                actions.append(action)

        prob = 1.0 / float( len( actions ) )
        actionDist = [( action, prob ) for action in actions]
        for action,prob in actionDist:
            successorPosition = game.Actions.getSuccessor(position,action)
            x ,y = map(int,successorPosition)

            dist[successorPosition] = prob

        return dist

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."

        noisyWalls = gameState.getNoisyWalls()

        self.observe(gameState, noisyWalls)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] >= 3 and neighbors(p,gameState) ]

        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, gameState, observationWalls):
        "Updates beliefs based on the given noisy perception of the walls and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over
        ghost locations conditioned on all evidence so far.
        """
        pass

class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm
    updates to compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over possible pacman positions.
        You may find useful self.legalPositions which contains the possible
        positions in the map.
        """
        #TODO: your code here
        util.raiseNotDefined()



    def observe(self, gameState, observationWalls):
        """
        Updates beliefs based on the perception of the current
        *unknown* position of the pacman.

        The observationWalls is a noisy perception of whether is a wall or not
        in the four adjacent locations (North,Sout,East,West). This amount of
        noise is given by an Epsilon (See busters.EPSILON).

        self.legalPositions is a list of the possible pacman positions (you
        should only consider positions that are in self.legalPositions).

        Remember that you *don't* know the pacman position and you must not
        make any direct assumptions about it, just what you can infer from the
        perceptions.

        """
        #TODO: your code here
        util.raiseNotDefined()




    def elapseTime(self, gameState):
        """
        Update self.beliefs in response to a time step passing from the current
        state.

        The model is updated to incorporate the knowledge acquired from the previous
        time. In order to obtain the distribution over new positions for the pacman,
        given its previous position (oldPos) use this line of code:

        newPostDist = self.getPacmanSuccesorDistribution(gameState,oldPos)

        Note that you may need to replace "oldPos" with the correct name of the
        variable that you have used to refer to the previous pacman position for
        which you are computing this distribution. You will need to compute
        multiple position distributions for a single update.

        newPosDist is a util.Counter object, where for each position p in
        self.legalPositions,

        newPostDist[p] = Pr( pacman is at position p at time t + 1 | pacman is at position oldPos at time t )
        """
        #TODO: Your code here:
        util.raiseNotDefined()

    def getBeliefDistribution(self):
        return self.beliefs

class ParticleFilter(InferenceModule):
    """
    A particle filter for finding a single ghost.

    Useful helper functions will include random.choice, which chooses
    an element from a list uniformly at random, and util.sample, which
    samples a key from a Counter by treating its values as probabilities.
    """


    def __init__(self, ghostAgent, numParticles=1000):
        InferenceModule.__init__(self, ghostAgent);

        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles


    def initializeUniformly(self, gameState):
        """
          Initializes a list of particles. Use self.numParticles for the number of particles.
          Use self.legalPositions for the legal board positions where a particle could be located.
          Particles should be evenly (not randomly) distributed across positions in order to
          ensure a uniform prior.

          ** NOTE **
            the variable you store your particles in must be a list; a list is simply a collection
            of unweighted variables (positions in this case). Storing your particles as a Counter or
            dictionary (where there could be an associated weight with each position) is incorrect
            and will produce errors
        """
        #TODO: your code here
        util.raiseNotDefined()

    def observe(self, gameState, observationWalls):
        """
        Update beliefs based on the given wall observation. Make sure to
        handle the special case where all particles have weight 0 after
        reweighting based on observation. If this happens, resample particles
        uniformly at random from the set of legal positions (self.legalPositions).

        Remember that when all particles receive 0 weight, they should be recreated from
        the prior distribution by calling initializeUniformly. The total
        weight for a belief distribution can be found by calling totalCount on a Counter object.

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.

        """
        #TODO: your code here
        util.raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

        newPostDist = self.getPacmanSuccesorDistribution(gameState,oldPos)

        to obtain the distribution over new positions for the pacman, given its
        previous position (oldPos).

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.
        """
        #TODO: your code here
        util.raiseNotDefined()

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over pacman
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        #TODO: your code here
        util.raiseNotDefined()
