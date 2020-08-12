# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, math
from game import Directions, Actions
import game
from util import nearestPoint


# 12312

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='FirstAgent', second='SecondAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class AgentBehaviour(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # start point
        self.start = gameState.getAgentPosition(self.index)
        self.isRed = gameState.isOnRedTeam(self.index)
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        if self.isRed:
            self.left = 0
            self.right = self.width // 2
            self.food_left = self.width // 2
            self.food_right = self.width
        else:
            self.left = self.width // 2
            self.right = self.width
            self.food_left = 0
            self.food_right = self.width // 2
        self.edges = self.getEdges(gameState)
        self.walls = gameState.getWalls()
        self.enemyFood = self.getFood(gameState).asList()
        self.myFood = self.getFoodYouAreDefending(gameState).asList()
        self.foodMissing = False
        self.oppoCaps = self.getCapsules(gameState)
        self.myCaps = self.getCapsulesYouAreDefending(gameState)
        self.opponents = self.getOpponents(gameState)
        self.myTeam = self.getTeam(gameState)[1]
        self.allCosts = self.analyseMap(gameState)
        self.mySafeFood = []
        self.TeamSafeFood = []
        self.agentPos = dict()
        self.agentState = dict()
        self.oppoGhosts = []
        self.dangerGhost = []
        self.oppoScareTimer = dict()

    def analyseMap(self, gameState):
        allCosts = dict()
        visited = []
        visited.append(self.start)
        q = util.Queue()
        q.push(self.start)
        while not q.isEmpty():
            current_Position = q.pop()
            for neighbor in Actions.getLegalNeighbors(current_Position, self.walls):
                if neighbor not in visited:
                    visited.append(neighbor)
                    q.push(neighbor)
                    if neighbor[0] in range(self.food_left, self.food_right):
                        direction = neighbor
                        if self.checkDeadEnd(current_Position, direction):
                            allCosts[current_Position] = (1, current_Position)
                            allCosts[direction] = (2, current_Position)
                            Q = util.Queue()
                            Q.push(direction)
                            while not Q.isEmpty():
                                suc = Q.pop()
                                for next in Actions.getLegalNeighbors(suc, self.walls):
                                    if next not in allCosts:
                                        allCosts[next] = (
                                            self.getMazeDistance(next, current_Position) + 1, current_Position)
                                        visited.append(next)
                                        Q.push(next)
                            continue
        return allCosts

    def checkDeadEnd(self, block, direction):
        target = self.start
        Q = util.PriorityQueue()
        Q.push(direction, 1)
        visited = [block, direction]
        while not Q.isEmpty():
            curPos = Q.pop()
            neighbors = Actions.getLegalNeighbors(curPos, self.walls)
            for neighbor in neighbors:
                if neighbor not in visited:
                    x, _ = neighbor
                    if (self.isRed and x <= self.right) or (not self.isRed and x >= self.left):
                        return False
                    visited.append(neighbor)
                    priority = self.getMazeDistance(neighbor, target)
                    Q.push(neighbor, priority)
        return True

    def getEdges(self, gameState):
        edges = []
        if self.isRed:
            oppoMid = self.right
            middle = oppoMid - 1
        else:
            middle = self.left
            oppoMid = middle - 1
        for y in range(self.height):
            if not gameState.hasWall(middle, y) and not gameState.hasWall(oppoMid, y):
                edges.append((middle, y))

        return edges

    def eatFood(self, targets, bothAttack):
        minDist = self.height * self.width
        minTarget = None
        for target in targets:
            dist = self.getMazeDistance(target, self.agentPos[self.index])
            if bothAttack == True:
                if self.priority == 1:
                    dist += self.height - target[1]
                else:
                    dist += target[1]

            if dist < minDist:
                minDist = dist
                minTarget = target

        return minTarget

    # the generate a new state; update all usefull info
    def updateGameState(self, gameState):
        # self.oppoPacSeen = []
        self.oppoGhosts = []
        self.oppoCaps = self.getCapsules(gameState)
        self.enemyFood = self.getFood(gameState).asList()
        self.dangerGhost = []

        for agent in range(gameState.getNumAgents()):
            self.agentState[agent] = gameState.getAgentState(agent)
            self.agentPos[agent] = gameState.getAgentPosition(agent)
            self.oppoScareTimer[agent] = self.agentState[agent].scaredTimer
            if agent in self.opponents:
                if not self.agentPos[agent] is None:
                    if self.agentState[agent].isPacman:
                        pass
                    else:
                        self.oppoGhosts += [agent]
                        if self.agentState[agent].scaredTimer <= 2:
                            self.dangerGhost += [agent]

        self.checkFoodMissing(gameState)
        self.TeamSafeFood = self.getSafeFood(self.myTeam)
        self.mySafeFood = self.getSafeFood(self.index)

    # This is the mean concept which calculated using allCosts
    # plus and minus the cost of myself and ghost accordingly if find in allCosts
    def getSafeFood(self, index):
        safeFood = []
        targets = list(self.enemyFood)
        targets += self.oppoCaps
        for food in targets:
            chaseCost = self.height * self.width
            chaser = None
            myCost = self.getMazeDistance(self.agentPos[index], food)
            for ghost in self.oppoGhosts:
                dist = self.getMazeDistance(self.agentPos[ghost], food)
                if dist < chaseCost:
                    chaseCost = dist
                    chaser = ghost
            if food in self.allCosts:
                cost, pt = self.allCosts[food]
                myCost += cost
                chaseCost -= cost
            if myCost < chaseCost or myCost < self.oppoScareTimer[chaser] - 1:
                safeFood.append(food)
        return safeFood

    def chooseOffense(self, gameState):

        a = False
        b = False
        if len(self.enemyFood) <= 2:
            return False, False
        self.mySafeFood += self.oppoCaps
        self.TeamSafeFood += self.oppoCaps
        a = len(self.mySafeFood) >= 0
        b = len(self.TeamSafeFood) >= 0
        if a and b:
            if len(self.enemyFood) > 2:
                return True, True

        if len(self.mySafeFood) == 0:
            a = False

        if len(self.TeamSafeFood) == 0:
            b = False

        return a, b

    def chooseAction(self, gameState):
        self.updateGameState(gameState)
        offense, mateOffense = self.chooseOffense(gameState)

        if offense:
            targets = []
            if len(self.enemyFood) > 2:
                targets = list(self.mySafeFood)
            if self.agentState[self.index].numCarrying > 2:
                targets += self.edges
            target = self.eatFood(targets, mateOffense)


        else:
            if self.agentState[self.index].numCarrying > 0:
                target = self.eatFood(self.edges, False)
            else:
                if mateOffense:
                    target = self.edges[len(self.edges) // 2]
                else:
                    if self.priority == 1:
                        target = self.edges[len(self.edges) // 3]

                    else:
                        target = self.edges[len(self.edges) * 2 // 3 - 1]

        if target is None:
            target = self.enemyFood[0]

        path = self.aStarSearch(gameState, target)
        if len(path) == 0:
            return Directions.STOP
        return path[0]

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def oppPac(self):
        oppoPac = []
        for i in self.opponents:
            if self.agentPos[i] and self.agentState[i].isPacman and self.agentState[i].scaredTimer <= 0:
                oppoPac.append(self.agentPos[i])
        return oppoPac

    def checkFoodMissing(self, gameState):
        preState = self.getPreviousObservation()
        if preState == None:
            return
        preFood_list = self.getFoodYouAreDefending(preState).asList()
        curFood_list = self.getFoodYouAreDefending(gameState).asList()
        for food in preFood_list:
            if food not in curFood_list:
                self.foodMissing = True
        return

    def aStarSearch(self, gameState, target):
        oppoGhost = set()
        for ghost in self.dangerGhost:
            oppoGhost.add(self.agentPos[ghost])
        if not self.getMazeDistance(self.agentPos[self.index], target) < 2 and not target in self.oppPac():
            for ghost in oppoGhost:
                oppoGhost = oppoGhost | set(Actions.getLegalNeighbors(ghost, self.walls))

        myPosition = gameState.getAgentPosition(self.index)
        exploring = util.PriorityQueue()
        exploring.push((gameState, myPosition, []), util.manhattanDistance(myPosition, target))
        visited = []
        visited.append(myPosition)
        while not exploring.isEmpty():
            currentState, currentPosition, path = exploring.pop()
            if currentPosition == target:
                return path
            else:
                for action in currentState.getLegalActions(self.index):
                    successor = self.getSuccessor(currentState, action)
                    successorPos = successor.getAgentPosition(self.index)
                    if successorPos not in visited:
                        if successorPos not in oppoGhost:
                            priority = util.manhattanDistance(target, successorPos) + len(path + [action])
                            exploring.push((successor, successorPos, path + [action]), priority)
                            if not target == successorPos:
                                visited.append(successorPos)
                        else:
                            continue
        return []


class FirstAgent(AgentBehaviour):
    def registerInitialState(self, gameState):
        AgentBehaviour.registerInitialState(self, gameState)
        self.priority = 1


class SecondAgent(AgentBehaviour):
    def registerInitialState(self, gameState):
        AgentBehaviour.registerInitialState(self, gameState)
        self.priority = 2


