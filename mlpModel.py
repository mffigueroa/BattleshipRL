import code
import os.path
import keras
from keras import regularizers
from keras.models import load_model
from LogOutputter import LogOutputter
import numpy as np
from ship import MoveOutcome
from vector2 import Vector2
from experienceReplayBuffer import ExperienceReplayBuffer
from mlpModel_build import BuildMLPModel
from rewardFunction import GetReward

class MLPAIModel:
	def __init__(self, playerNumber, logOutputter=None, outputDiagnostics=None):
		self.stateSeq = []
		self.stateSeqVectors = []
		self.playerNumber = playerNumber
		self.modelName = 'MLP Model v1'
		
		self.modelSavePrefix = '{}_Player{}'.format(self.modelName.replace(' ','_'),  self.playerNumber)
		self.actorModelFilename = '{}_actor.h5'.format(self.modelSavePrefix)
		self.criticModelFilename = '{}_critic.h5'.format(self.modelSavePrefix)
		
		self.logOutputter = logOutputter
		
		self.outputDiagnostics = outputDiagnostics
		if not self.outputDiagnostics is None:
			self.diagnosticsLogOutputter = LogOutputter(self.logOutputter.logLocation + '_qValues.txt')
			self.diagnosticsStateAtOutput = set([])
		else:
			self.diagnosticsLogOutputter = None
			self.diagnosticsStateAtOutput = None		
		
		self.modelIterations = 0		
		self.trainCriticEveryIter = 20
		self.saveModelEveryIter = 100
		self.experienceBufferSize = 999
		self.explorationProb = 0.005
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
		self.maxSeqLength = 1
		self.inputDimension = self.normedBoardPositions * self.maxMoveOutcome + 2
		self.outputDimension = self.normedBoardPositions
		
		if os.path.isfile(self.actorModelFilename):
			print('Loading {}...'.format(self.actorModelFilename))
			self.actorModel = load_model(self.actorModelFilename)
		else:
			self.actorModel = BuildMLPModel(self.maxSeqLength, self.inputDimension, self.outputDimension)
		
		if os.path.isfile(self.criticModelFilename):
			print('Loading {}...'.format(self.criticModelFilename))
			self.criticModel = load_model(self.criticModelFilename)
		else:
			l2Regularizer = regularizers.l2(0.01)
			biggerl2Regularizer = regularizers.l2(0.05)
			self.criticModel = BuildMLPModel(self.maxSeqLength, self.inputDimension, self.outputDimension, kernel_l2Reg=l2Regularizer, bias_l2Reg=l2Regularizer, last_kernel_l2Reg=biggerl2Regularizer, last_bias_l2Reg=biggerl2Regularizer)
		
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
		self.logOutputter.Output('\n\n\n\n\nClearing MLP model state\n\n\n\n\n')
		
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
			raise Exception('MLPAIModel CreateStateVector called on invalid state data.')
		self.SetActualBoardDimensions(state.moveOutcomes.shape)
		moveOutcomeVector = self.CreateMoveOutcomesVector(state.moveOutcomes)
		moveVectorUniqueValues = set(moveOutcomeVector.flatten().tolist())
		filteredValues = set([0,1])
		if len(moveVectorUniqueValues.difference(filteredValues)) > 0:
			code.interact(local=locals())
		
		numericalStateVector = np.array([state.aliveShips, state.piecesBeenHit])
		stateVector = np.append(moveOutcomeVector, numericalStateVector)
		return stateVector
		
	def AppendState(self, state):		
		if len(self.stateSeqVectors) > self.maxSeqLength and len(self.stateSeqVectors) > 1:
			self.stateSeqVectors = self.stateSeqVectors[1:]
			self.stateSeq = self.stateSeq[1:]
		self.stateSeq.append(state)
		stateVec = self.CreateStateVector(state)
		self.stateSeqVectors.append(stateVec)
		
	def GetCriticBellmanDifference(self, statesAfterMove, moves, rewards):
		batchIndices = np.arange(len(moves))
		actorOutput = self.RunModelAtStates(statesAfterMove, critic=False)
		criticOutput = self.RunModelAtStates(statesAfterMove, critic=True)
		actorBestMoves = np.argmax(actorOutput, axis=-1)
		maxQValuesAtState = criticOutput[batchIndices,actorBestMoves]
		return rewards + 0.99*maxQValuesAtState
	
	def TrainOnExperiences(self, states, statesAfterMove, moves, rewards, actorOutputs, critic=None):
		if critic is None:
			critic = False
		batchIndices = np.arange(len(moves))
		actorOutputs[batchIndices,moves] = self.GetCriticBellmanDifference(statesAfterMove, moves, rewards)
		return self.TrainModel(states, actorOutputs, critic)
		
	def ReceiveStateUpdate(self, state):
		self.actualBoardDimensions = state.moveOutcomes.shape
		self.AppendState(state)
		ownShipsDestroyed = 0
		stateOfRounds = [state.moveOutcomes]
		if len(self.stateSeq) > 1:
			stateOfRounds.append(self.stateSeq[-2].moveOutcomes)
			ownShipsDestroyed = self.stateSeq[-2].aliveShips - state.aliveShips
		
		if not self.stateBeforeLastMove is None and not self.lastModelOutput is None and not self.lastModelMove is None:
			numMovesAtPosition = 0
			if self.lastModelMove in self.numModelMovesAt:
				numMovesAtPosition = self.numModelMovesAt[self.lastModelMove]
				self.numModelMovesAt[self.lastModelMove] += 1
			else:
				self.numModelMovesAt[self.lastModelMove] = 1
			
			# calculate reward
			moveOutcomesOfRound = GetRoundMoveOutcomeVector(stateOfRounds)
			for outcomeIndex in range(1,len(moveOutcomesOfRound)):
				if moveOutcomesOfRound[outcomeIndex] > 1:
					code.interact(local=locals())
			reward = GetReward(moveOutcomesOfRound, ownShipsDestroyed, numMovesAtPosition, self.logOutputter)
			self.rewards.append(reward)
			
			stateAfterMove = self.GetModelInputForState(self.stateSeqVectors)
			
			if not self.outputDiagnostics is None:
				stateHash = tuple(self.stateBeforeLastMove.flatten().tolist())
				if not stateHash in self.diagnosticsStateAtOutput:
					self.diagnosticsStateAtOutput.add(stateHash)
					self.diagnosticsLogOutputter.Output('Input')
					self.diagnosticsLogOutputter.Output(str(self.stateBeforeLastMove))
					self.diagnosticsLogOutputter.Output('')
					self.diagnosticsLogOutputter.Output('Reward')
					self.diagnosticsLogOutputter.Output(str(reward))
					self.diagnosticsLogOutputter.Output('')
					self.diagnosticsLogOutputter.Output('Output')
					self.diagnosticsLogOutputter.Output(str(self.lastModelOutput))
					self.diagnosticsLogOutputter.Output('')
					self.diagnosticsLogOutputter.Output('')
			
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
			self.experienceBuffer.Insert(self.stateBeforeLastMove[0,:], stateAfterMove[0,:], self.lastModelMove, reward)
			self.modelIterations += 1
			
			if self.modelIterations % self.trainCriticEveryIter == 0:
				self.criticModel.set_weights(self.actorModel.get_weights())
			if self.modelIterations > 0 and self.modelIterations % self.saveModelEveryIter == 0:
				try:
					self.criticModel.save(self.criticModelFilename)
					self.actorModel.save(self.actorModelFilename)
				except:
					pass
			
			self.stateBeforeLastMove = None
			self.lastModelOutput = None
			self.lastModelMove = None
	
	def GetModelInputForState(self, stateVectors):
		modelInput = np.zeros((1, self.maxSeqLength, self.inputDimension))
		for seqNum in range(self.maxSeqLength):
			modelInput[0, seqNum, :] = stateVectors[seqNum]
		modelInput = modelInput.reshape((1, self.maxSeqLength * self.inputDimension))
		return modelInput
	
	def RunModelAtStates(self, modelInputs, critic=None):
		if critic is None or not critic:
			modelOutput = self.actorModel.predict(modelInputs, batch_size=len(modelInputs), verbose=0)
		else:
			modelOutput = self.criticModel.predict(modelInputs, batch_size=len(modelInputs), verbose=0)
		if modelOutput is None or modelOutput.shape != (len(modelInputs), self.outputDimension):
			raise Exception('MLP Model predict returned invalid result.')
		return modelOutput
	
	def TrainModel(self, modelInput, correctModelOutput, critic=None):
		if critic is None or not critic:
			fitResult = self.actorModel.fit(modelInput, correctModelOutput, batch_size=modelInput.shape[0], epochs=1, verbose=0)
		else:
			fitResult = self.criticModel.fit(modelInput, correctModelOutput, batch_size=modelInput.shape[0], epochs=1, verbose=0)
		OutputModelMetrics(fitResult, self.logOutputter)
	
	def GetMovesFromModelOutput(self, modelOutputs, exploreRandomly=None):
		movesSortedByQValue = np.argsort(-modelOutputs, axis=-1)
		modelMoves = []
		topKToPick = np.random.randint(0,2)
		
		for rowNum in range(movesSortedByQValue.shape[0]):
			currentTopK = 0
			if exploreRandomly and np.random.rand(1) <= self.explorationProb:
				modelMoves.append(np.random.randint(0,self.normedBoardPositions))
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
		self.logOutputter.Output('MLP shooting at {}'.format(moveVec))
		return moveVec

def GetRoundMoveOutcomeVector(stateOfRounds):
	moveOutcomesOfRound = np.zeros((len(MoveOutcome)))
	
	if len(stateOfRounds) > 1 and stateOfRounds[0].shape != stateOfRounds[1].shape:
		raise Exception('MLPAIModel GetRoundMoveOutcomeVector rounds must have the same shape.')
	
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