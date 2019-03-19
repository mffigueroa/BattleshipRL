import code
import os.path
import keras
from keras.models import Sequential, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed
import numpy as np
from ship import MoveOutcome
from vector2 import Vector2
from experienceReplayBuffer import ExperienceReplayBuffer

class LSTMAIModel:
	def __init__(self, playerNumber, logOutputter=None):
		self.stateSeq = []
		self.stateSeqVectors = []
		self.playerNumber = playerNumber
		self.modelName = 'LSTM Model v1'
		
		self.modelSavePrefix = '{}_Player{}'.format(self.modelName.replace(' ','_'),  self.playerNumber)
		self.actorModelFilename = '{}_actor.h5'.format(self.modelSavePrefix)
		self.criticModelFilename = '{}_critic.h5'.format(self.modelSavePrefix)
		
		self.logOutputter = logOutputter
		self.modelIterations = 0
		
		self.trainCriticEveryIter = 20
		self.saveModelEveryIter = 100
		self.experienceBufferSize = 99
		self.explorationProb = 0.1
		self.experienceBuffer = ExperienceReplayBuffer(self.experienceBufferSize)
		self.rewards = []
		
		self.normedBoardLength = 10
		self.normedBoardPositions = self.normedBoardLength*self.normedBoardLength
		self.maxMoveOutcome = len(MoveOutcome) + 1 # add 1 for empty hit result
		
		self.actualBoardDimensions = None
		
		# input:
		#	10x10 board max x 5+1 hit states per position
		#	alive ships
		#	piecesBeenHit
		# output:
		#	10x10 board positions
		self.maxSeqLength = 10
		self.inputDimension = self.normedBoardPositions * self.maxMoveOutcome + 2
		self.outputDimension = self.normedBoardPositions
		
		if os.path.isfile(self.actorModelFilename):
			print('Loading {}...'.format(self.actorModelFilename))
			self.actorModel = load_model(self.actorModelFilename)
		else:
			self.actorModel = BuildLSTMModel(self.maxSeqLength, self.inputDimension, self.outputDimension)
		
		if os.path.isfile(self.criticModelFilename):
			print('Loading {}...'.format(self.criticModelFilename))
			self.criticModel = load_model(self.criticModelFilename)
		else:
			self.criticModel = BuildLSTMModel(self.maxSeqLength, self.inputDimension, self.outputDimension)
		
		self.numModelMovesAt = {}
		self.stateBeforeLastMove = None
		self.lastModelOutput = None
		self.lastModelMove = None
	
	def GetPlayerNumber(self):
		return self.playerNumber
	
	def ClearState(self):
		self.stateSeq = []
		self.stateSeqVectors = []
		self.rewards = []
		self.numModelMovesAt = {}
		self.stateBeforeLastMove = None
		self.lastModelOutput = None
		self.lastModelMove = None
		self.logOutputter.Output('\n\n\n\n\nClearing LSTM model state\n\n\n\n\n')
		
	def GetModelName(self):
		return self.modelName
	
	def NormalizeBoardPosition(self, row, col):
		normalizedRow = int((float(row) / self.actualBoardDimensions[1]) * self.normedBoardLength)
		normalizedCol = int((float(col) / self.actualBoardDimensions[0]) * self.normedBoardLength)
		return normalizedRow, normalizedCol
		
	def UnnormalizeBoardPosition(self, boardPos):
		normedRow = boardPos / self.normedBoardLength
		normedCol = boardPos - int(normedRow) * self.normedBoardLength
		row = int((normedRow / self.normedBoardLength) * self.actualBoardDimensions[1])
		col = int((normedCol / self.normedBoardLength) * self.actualBoardDimensions[0])
		
		if row < 0 or row > 9 or col < 0 or col > 9:
			print('UnnormalizeBoardPosition {}: {}, {}'.format(boardPos, row, col))
		return row, col
	
	def SetActualBoardDimensions(self, dimensions):
		self.actualBoardHeight = dimensions[0]
		self.actualBoardWidth = dimensions[1]
	
	def CreateMoveOutcomesVector(self, moveOutcomes):
		normalizedMoveOutcomes = np.zeros((self.normedBoardLength,self.normedBoardLength,self.maxMoveOutcome))
		
		for rowNumber in range(moveOutcomes.shape[0]):
			for colNumber in range(moveOutcomes.shape[1]):
				moveOutcome = int(moveOutcomes[rowNumber,colNumber])
				normalizedRow, normalizedCol = self.NormalizeBoardPosition(rowNumber, colNumber)
				normalizedMoveOutcomes[normalizedRow,normalizedCol,moveOutcome] = 1		
		return np.array(normalizedMoveOutcomes).flatten()
	
	def CreateStateVector(self, state):
		if state is None or state.moveOutcomes is None or state.moveOutcomes.shape[0] < 1:
			raise Exception('LSTMAIModel CreateStateVector called on invalid state data.')
		self.SetActualBoardDimensions(state.moveOutcomes.shape)
		moveOutcomeVector = self.CreateMoveOutcomesVector(state.moveOutcomes)
		numericalStateVector = np.array([state.aliveShips, state.piecesBeenHit])
		stateVector = np.append(moveOutcomeVector, numericalStateVector)
		return stateVector
		
	def AppendState(self, state):		
		if len(self.stateSeqVectors) >= self.maxSeqLength - 1:
			self.stateSeqVectors = self.stateSeqVectors[1:]
			self.stateSeq = self.stateSeq[1:]
		self.stateSeq.append(state)
		stateVec = self.CreateStateVector(state)
		self.stateSeqVectors.append(stateVec)
		
	def GetCriticBellmanDifference(self, statesAfterMove, moves, rewards):
		batchIndices = np.arange(len(moves))
		criticOutput = self.RunModelAtStates(statesAfterMove, critic=True)
		maxQValuesAtState = np.max(criticOutput, axis=-1)
		return rewards + 0.99*maxQValuesAtState
	
	def TrainOnExperiences(self, states, statesAfterMove, moves, rewards, actorOutputs, critic=None):
		if critic is None:
			critic = False
		batchIndices = np.arange(len(moves))
		self.logOutputter.Output('Model predicted Q-values: {}'.format(actorOutputs[batchIndices,moves]))
		actorOutputs[batchIndices,moves] = self.GetCriticBellmanDifference(statesAfterMove, moves, rewards)
		self.logOutputter.Output('Model target Q-values: {}'.format(actorOutputs[batchIndices,moves]))
		return self.TrainModel(states, actorOutputs, critic)
		
	def ReceiveStateUpdate(self, state):
		self.actualBoardDimensions = state.moveOutcomes.shape
		self.AppendState(state)
		ownShipsDestroyed = 0		
		stateOfRounds = [state.moveOutcomes]
		if len(self.stateSeq) > 1:
			stateOfRounds.append(self.stateSeq[-2].moveOutcomes)
			ownShipsDestroyed = self.stateSeq[-2].aliveShips - state.aliveShips
		
		if not self.stateBeforeLastMove is None and not self.lastModelOutput is None and not	self.lastModelMove is None:
			numMovesAtPosition = 0
			if self.lastModelMove in self.numModelMovesAt:
				numMovesAtPosition = self.numModelMovesAt[self.lastModelMove]
				self.numModelMovesAt[self.lastModelMove] += 1
			else:
				self.numModelMovesAt[self.lastModelMove] = 1
			
			# calculate reward
			moveOutcomesOfRound = GetMoveOutcomesOfRound(stateOfRounds)
			reward = GetReward(moveOutcomesOfRound, ownShipsDestroyed, numMovesAtPosition, self.logOutputter)
			self.rewards.append(reward)
			
			stateAfterMove = self.GetModelInputForState(self.stateSeqVectors)
			
			# build batch from current state and experiences buffer
			experiences_states, experiences_statesAfterMove, experiences_moves, experiences_rewards = self.experienceBuffer.GetBatch().ToMatrices()
			if experiences_states.shape[0] > 0:
				actorOutputsAtExperiences = self.RunModelAtStates(experiences_states)
				
				all_states = np.append(experiences_states, stateAfterMove, axis=0)
				all_statesAfterMove = np.append(experiences_statesAfterMove, self.stateBeforeLastMove, axis=0)
				all_moves = np.append(experiences_moves, [self.lastModelMove])
				all_rewards = np.append(experiences_rewards, [reward])
				all_actorOutputs = np.append(actorOutputsAtExperiences, self.lastModelOutput, axis=0)
			else:
				all_states = stateAfterMove
				all_statesAfterMove = self.stateBeforeLastMove
				all_moves = np.array([self.lastModelMove])
				all_rewards = np.array([reward])
				all_actorOutputs = self.lastModelOutput
			
			# train model and add new experience to buffer
			self.TrainOnExperiences(all_states, all_statesAfterMove, all_moves, all_rewards, all_actorOutputs)
			self.experienceBuffer.Insert(self.stateBeforeLastMove[0,:,:], stateAfterMove[0,:,:], self.lastModelMove, reward)
			self.modelIterations += 1
			
			if self.modelIterations % self.trainCriticEveryIter == 0:
				self.logOutputter.Output('Training critic\n\n')
				print('\nTraining critic\n')
				criticOutputsAtExperiences = self.RunModelAtStates(all_states, critic=True)
				self.TrainOnExperiences(all_states, all_statesAfterMove, all_moves, all_rewards, criticOutputsAtExperiences, critic=True)
			if self.modelIterations > 0 and self.modelIterations % self.saveModelEveryIter == 0:
				try:
					self.criticModel.save(self.criticModelFilename)
					self.actorModel.save(self.actorModelFilename)
					print('Saved {} models'.format(self.modelSavePrefix))
				except:
					pass
			
			self.stateBeforeLastMove = None
			self.lastModelOutput = None
			self.lastModelMove = None
	
	def GetModelInputForState(self, stateVectors):
		modelInput = np.zeros((1, self.maxSeqLength, self.inputDimension))
		for seqNum in range(len(stateVectors)):
			modelInput[0, seqNum, :] = stateVectors[seqNum]
		return modelInput
	
	def RunModelAtStates(self, modelInputs, critic=None):
		if critic is None or not critic:
			modelOutput = self.actorModel.predict(modelInputs, batch_size=len(modelInputs), verbose=1)
		else:
			modelOutput = self.criticModel.predict(modelInputs, batch_size=len(modelInputs), verbose=1)
		if modelOutput is None or modelOutput.shape != (len(modelInputs), self.outputDimension):
			raise Exception('LSTM Model predict returned invalid result.')
		return modelOutput
	
	def TrainModel(self, modelInput, correctModelOutput, critic=None):
		if critic is None or not critic:
			fitResult = self.actorModel.fit(modelInput, correctModelOutput, batch_size=modelInput.shape[0], epochs=15, verbose=1)
		else:
			fitResult = self.criticModel.fit(modelInput, correctModelOutput, batch_size=modelInput.shape[0], epochs=50, verbose=1)
		OutputModelMetrics(fitResult, self.logOutputter)
	
	def GetMovesFromModelOutput(self, modelOutputs, exploreRandomly=None):
		movesSortedByQValue = np.argsort(modelOutputs, axis=-1)
		modelMoves = []		
		topKToPick = np.random.randint(0,4)
		
		for rowNum in range(movesSortedByQValue.shape[0]):
			numTries = 0
			currentTopK = 0
			if exploreRandomly and np.random.rand(1) <= self.explorationProb:
				modelMoves.append(np.random.randint(0,self.normedBoardPositions))
				self.logOutputter.Output('Making random move...')
				continue
			for moveNumIndex in range(movesSortedByQValue.shape[1]):
				moveNum = movesSortedByQValue[rowNum, moveNumIndex]
				if moveNumIndex + 1 >= movesSortedByQValue.shape[1]:
					if exploreRandomly:
						modelMoves.append(np.random.randint(0,self.normedBoardPositions))
					else:
						modelMoves.append(moveNum)
					break
				elif not moveNum in self.numModelMovesAt and currentTopK == topKToPick:
					self.logOutputter.Output('Picking top {}: {}'.format(topKToPick, moveNum))
					modelMoves.append(moveNum)
					break
				else:
					currentTopK += 1
		boardPositions = np.array(modelMoves)
		return boardPositions
		
	def GetNextMove(self):
		modelInput = self.GetModelInputForState(self.stateSeqVectors)
		modelOutput = self.RunModelAtStates(modelInput)
		boardPos = self.GetMovesFromModelOutput(modelOutput, exploreRandomly=True)
		if len(boardPos.shape) > 0:
			boardPos = boardPos.flatten()[0]
		else:
			boardPos = int(boardPos)
		
		self.stateBeforeLastMove = modelInput
		self.lastModelOutput = modelOutput
		self.lastModelMove = boardPos
			
		moveRow, moveCol = self.UnnormalizeBoardPosition(boardPos)
		moveVec = Vector2(moveRow, moveCol)
		self.logOutputter.Output('LSTM shooting at {}'.format(moveVec))
		return moveVec

def BuildLSTMModel(maxSeqLength, inputDimension, outputDimension, lstmUnits = None):
	if lstmUnits is None:
		lstmUnits = 8
	model = Sequential()
	model.add(LSTM(lstmUnits, return_sequences=True, input_shape=(maxSeqLength, inputDimension), recurrent_initializer='glorot_uniform', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.add(LSTM(lstmUnits, input_shape=(maxSeqLength, inputDimension), recurrent_initializer='glorot_uniform', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.add(Dense(outputDimension, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','acc'])
	return model

class RewardState:
	def __init__(self):
		self.opponentShipsDestroyed = 0
		self.ownShipsDestroyed = 0
		self.successfulHits = 0
		self.hitsOnDeadShip = 0
		self.redundantHits = 0
		self.misses = 0
		self.numMovesAtPosition = 0
		
	def OutputToLog(self, logOutputter):
		logOutputter.Output('ownShipsDestroyed: {}'.format(self.ownShipsDestroyed))
		logOutputter.Output('opponentShipsDestroyed: {}'.format(self.opponentShipsDestroyed))
		logOutputter.Output('successfulHits: {}'.format(self.successfulHits))
		logOutputter.Output('misses: {}'.format(self.misses))
		logOutputter.Output('hitsOnDeadShip: {}'.format(self.hitsOnDeadShip))
		logOutputter.Output('redundantHits: {}'.format(self.redundantHits))
		logOutputter.Output('numMovesAtPosition: {}'.format(self.numMovesAtPosition))
	
def GetReward(moveOutcomesOfRound, ownShipsDestroyed, numMovesAtPosition, logOutputter):
	rewardState = RewardState()
	rewardState.opponentShipsDestroyed = moveOutcomesOfRound[MoveOutcome.DestroyedShip.value]
	rewardState.successfulHits = moveOutcomesOfRound[MoveOutcome.HitAliveShip.value]
	rewardState.misses = moveOutcomesOfRound[MoveOutcome.Miss.value]
	rewardState.hitsOnDeadShip = moveOutcomesOfRound[MoveOutcome.HitAlreadyDestroyedShip.value]
	rewardState.redundantHits = moveOutcomesOfRound[MoveOutcome.HitShipWhereAlreadyHit.value]
	rewardState.numMovesAtPosition = numMovesAtPosition
	rewardState.ownShipsDestroyed = ownShipsDestroyed
	rewardState.OutputToLog(logOutputter)
	
	reward = rewardState.opponentShipsDestroyed * 20
	reward += rewardState.successfulHits * 10
	
	#reward -= rewardState.ownShipsDestroyed * 3
	#reward -= min(rewardState.hitsOnDeadShip, 5) * 2
	#reward -= min(rewardState.redundantHits, 5) * 2
	reward -= min(rewardState.misses, 5)
	reward -= min(rewardState.numMovesAtPosition, 4) * 3
	
	reward = max(reward, -20)
	
	logOutputter.Output('Reward: {}'.format(reward))
	
	return reward
		
def GetMoveOutcomesOfRound(stateOfRounds):
	moveOutcomesOfRound = np.zeros((len(MoveOutcome)))
	
	if len(stateOfRounds) > 1 and stateOfRounds[0].shape != stateOfRounds[1].shape:
		raise Exception('LSTMAIModel GetMoveOutcomesOfRound rounds must have the same shape.')
	
	rows = stateOfRounds[0].shape[0]
	columns = stateOfRounds[0].shape[1]
	
	for row in range(rows):
		for col in range(rows):
			round0_result = int(stateOfRounds[0][row,col])
			if len(stateOfRounds) < 2 or round0_result != int(stateOfRounds[1][row,col]):
				moveOutcomesOfRound[round0_result] += 1
	return moveOutcomesOfRound

def OutputModelMetrics(fitResult, logOutputter):
	if not fitResult is None:
		metrics = fitResult.history
		for name, values in metrics.items():
			logOutputter.Output('{}: {}'.format(name, values[-1]))
		logOutputter.Output('\n\n')