from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.goalPosition = None

        self.useAstar = False
        self.alhpa = 0.5
        self.discountFactor = 0.9
        self.weights = {'carrying': 0.0, 'successorScore': 48.255325264985274, 'run': 0.003414526409853335, 'distanceToFood': -1.3584191993966284, 'back': 0.1770267484677373, 'eatGhost': -0.03556763418234325}

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start,pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def getMiddleLines(self,gameState):
        if self.red:
            middleLine = [((gameState.data.layout.width / 2) - 1, y) for y in range(0, gameState.data.layout.height)]
        else:
            middleLine = [(gameState.data.layout.width / 2 + 1, y) for y in range(0, gameState.data.layout.height)]
        availableMiddle = [a for a in middleLine if a not in gameState.getWalls().asList()]
        return availableMiddle

    def astar(self, gameState, goalPosition):
        open_set = util.PriorityQueue()
        closed = []
        cost = 0
        open_set.push((gameState,[]), 0)
        food_list = self.getFood(gameState).asList()
        while not open_set.isEmpty():
            current_state, path = open_set.pop()
            current_position = current_state.getAgentPosition(self.index)

            if current_position == goalPosition:
                return path[0]

            #if (current_position in food_list):
            #    return path[0]

            if current_position in closed:
                continue
            else:
                closed.append(current_position)
                actions = current_state.getLegalActions(self.index)

            for action in actions:
                successor = current_state.generateSuccessor(self.index, action)
                successor_position = successor.getAgentPosition(self.index)
                cost = cost + self.getMazeDistance(successor_position, goalPosition)
                new_path = path + [action]
                open_set.push([successor, new_path], cost)

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        successorPosition = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()
        mapsize = gameState.getWalls().width * gameState.getWalls().height
        currentCarry = gameState.getAgentState(self.index).numCarrying
        middline = self.getMiddleLines(gameState)

        distanceToMid = min([self.getMazeDistance(successorPosition,midposition) for midposition in middline])
        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(successorPosition, food) for food in foodList])
            features['distanceToFood'] = float(minDistance) /mapsize

        features['run'] = 0.0

        if gameState.isOnRedTeam(self.index):
            #red team
            redFood = gameState.getRedFood().asList()
            if len(redFood) > 0:
                features['successorScore'] = -float(len(foodList))/len(redFood)
        else:
            #blue team
            blueFood = gameState.getBlueFood().asList()
            if len(blueFood) > 0:
                features['successorScore'] = -float(len(foodList))/len(blueFood)

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        ghostsPosition = [ghost.getPosition() for ghost in ghosts]
        if successor.getAgentPosition(self.index) == self.start:
            if gameState.getAgentState(self.index).isPacman :
                if currentCarry != 0:
                    features['back'] = -float(distanceToMid)/mapsize

        if len(ghostsPosition) >0:
            if min([self.getMazeDistance(successorPosition, ghostPos) for ghostPos in ghostsPosition]) <5:
                features['back'] = -float(distanceToMid)/mapsize
                features['run'] = -float(min(self.getMazeDistance(successorPosition, ghostPos) for ghostPos in ghostsPosition))/mapsize

        if len(foodList) <=2 or currentCarry >= 6:
            features['distanceToFood'] = 0
            features['successorScore'] = 0
            features['back'] = -float(distanceToMid)/mapsize
        features.divideAll(10.0)

        return features

    def getFeatures1(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        walls = gameState.getWalls()
        successorPosition = successor.getAgentState(self.index).getPosition()
        InitialPosition = gameState.getInitialAgentPosition(self.index)
        currentCarry = gameState.getAgentState(self.index).numCarrying

        border = self.getMiddleLines(gameState)
        distanceToBorder = min([self.getMazeDistance(successorPosition, borderPos) for borderPos in border])

        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(successorPosition, food) for food in foodList])
            features['distanceToFood'] = float(minDistance) /(walls.width * walls.height)

        features['enemyOneStepToPacman'] = 0.0
        blueFood = gameState.getBlueFood().asList()
        redFood = gameState.getRedFood().asList()
        if gameState.isOnRedTeam(self.index):
            if len(blueFood) != 0:
                features['successorScore'] = -float(len(foodList))/len(blueFood)
        else:
            if len(redFood) != 0:
                features['successorScore'] = -float(len(foodList))/len(redFood)

        enemies = []
        for opponent in self.getOpponents(gameState):
            enemy = gameState.getAgentState(opponent)
            enemies.append(enemy)
        enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        enemyGhostPosition = [Ghost.getPosition() for Ghost in enemyGhost]

        if successor.getAgentPosition(self.index) == InitialPosition:
                if gameState.getAgentState(self.index).isPacman :
                    if currentCarry != 0:
                        features['back'] = -float(distanceToBorder)/(walls.width * walls.height)

        if len(enemyGhostPosition) >0:
                if min([self.getMazeDistance(successorPosition, ghostPos) for ghostPos in enemyGhostPosition]) <5:
                        features['back'] = -float(distanceToBorder)/(walls.width * walls.height)
                        features['enemyOneStepToPacman'] = -float(min(self.getMazeDistance(successorPosition, ghostPos) for ghostPos in enemyGhostPosition))/(walls.width * walls.height)



        if len(foodList) <=2 or currentCarry >= 6:
            features['distanceToFood'] = 0
            features['successorScore'] = 0
            features['back'] = -float(distanceToBorder)/(walls.width * walls.height)
        features.divideAll(10.0)
        return features


    def chooseAction(self,gameState):
        actions = gameState.getLegalActions(self.index)

        if gameState.isOnRedTeam(self.index):
            midX = (gameState.data.layout.width / 2) - 1
        else:
            midX = (gameState.data.layout.width / 2) + 1
        middline = self.getMiddleLines(gameState)

        currentPosition = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        ghostsPosition = [ghost.getPosition() for ghost in ghosts]

        if len(ghostsPosition) >0 and not self.useAstar:
            distanceToEnemyGhost = (min([self.getMazeDistance(currentPosition, ghostPos) for ghostPos in ghostsPosition]))
            if distanceToEnemyGhost < 5 and gameState.getAgentState(self.index).isPacman:
                randomEntry = random.choice(middline)
                while randomEntry == currentPosition:
                    randomEntry = random.choice(middline)
                self.goalPosition = randomEntry
                self.useAstar = True


        if self.goalPosition != currentPosition:
            if self.useAstar:
                action = self.astar(gameState, self.goalPosition)
                return action
        else:
            self.useAstar = False

        action = self.getPolicy(gameState)

        return action



    def getPolicy(self, gameState):
        value, action = self.getMaxQWithAction(gameState)

        self.weights = self.update(gameState, action)
        return action

    def getQ(self, gameState, action): # get Q-value(Approximate Q-function Computation)
        features = self.getFeatures(gameState, action)
        qValue = 0
        for feature in features:
            qValue = qValue + features[feature] * self.weights[feature]
        return qValue

    def getMaxQWithAction(self, gameState): # get value and get policy
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        qValues = []
        for action in actions:
            qValues.append((self.getQ(gameState, action), action))
        value, policy = max(qValues)
        #self.update(gameState, policy)
        return value, policy

    def getReward(self, gameState, action):
        if self.getPreviousObservation() is None:
            return 0

        successor = self.getSuccessor(gameState, action)
        myCurrentPos = gameState.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        ghostsPosition = [ghost.getPosition() for ghost in ghosts]
        if len(ghostsPosition) >0:
            distanceToEnemyGhost = (min([self.getMazeDistance(myCurrentPos, ghostPos) for ghostPos in ghostsPosition]))
        else:
            distanceToEnemyGhost = 9999

        nextx, nexty = successor.getAgentPosition(self.index)

        if gameState.hasFood(nextx,nexty):
            wallCount = 0
            if gameState.hasWall(nextx+1,nexty):
                wallCount +=1
            if gameState.hasWall(nextx-1,nexty):
                wallCount += 1
            if gameState.hasWall(nextx,nexty+1):
                wallCount += 1
            if gameState.hasWall(nextx,nexty-1):
                wallCount += 1
            if wallCount>=3 and distanceToEnemyGhost <=2:
                reward = -1
            else:
                reward = 1
        else:
            reward = -1

        if self.getScore(successor) - self.getScore(gameState)<=0:
            reward -= 1
        else:
            reward +=1

        return reward

    def update(self, gameState, action): #update weights(need to update for each feature i for the last executed actiona.)
        reward = self.getReward(gameState, action)
        successor = self.getSuccessor(gameState, action)
        currentQ = self.getQ(gameState, action)
        weights = self.weights
        nextQ, nextPolicy = self.getMaxQWithAction(successor)
        features = self.getFeatures(gameState, action)
        for feature in self.getFeatures(gameState, action):
            weights[feature] = weights[feature] + (self.alhpa * (reward + self.discountFactor * nextQ - currentQ))*features[feature]
        return weights

    def final(self, gameState):
        #print self.weights
        file = open('weights1.txt', 'w')
        file.write(str(self.weights))

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.foodMissingPos = None
        self.findFoodMissing = False

    def getInvaders(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaders)==0:
            return None
        else:
            return invaders

    def chooseAction(self, gameState):
        invaders = self.getInvaders(gameState)
        middleLines = self.getMiddleLines(gameState)
        len(middleLines)
        middle = middleLines[len(middleLines)//2]
        self.checkFoodMissing(gameState)
        if self.findFoodMissing and invaders == None:
            action = self.astar(gameState, self.foodMissingPos)
            return action

        if invaders != None:
            invadersDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for a in invaders]
            minDIs = min(invadersDis)
            target = [a.getPosition() for a, v in zip(invaders, invadersDis) if v == minDIs]
            return self.astar(gameState,target[0])

        if gameState.getAgentState(self.index).getPosition()==middle or gameState.getAgentState(self.index).getPosition() == self.foodMissingPos:
            return self.astar(gameState,self.start)

        return self.astar(gameState,middle)

    def checkFoodMissing(self,gameState):
        self.findFoodMissing = False
        previState = self.getPreviousObservation()
        if previState == None:
            return
        previFood_list = self.getFoodYouAreDefending(previState).asList() #根据每一次当前状态和之前状态比较得出当时丢失的点，就算被对方吃了回到了自己那边也可以用
        curreFood_list = self.getFoodYouAreDefending(gameState).asList()
        for food in previFood_list:
            if food not in curreFood_list:
                self.foodMissingPos = food
                self.findFoodMissing = True
        if gameState.getAgentPosition(self.index) == self.foodMissingPos:
            self.findFoodMissing = False
        return

