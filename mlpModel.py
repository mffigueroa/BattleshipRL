import code
import time
import os.path
import keras
import keras.backend as K
import tensorflow as tf
from keras import regularizers
from keras.models import load_model
from LogOutputter import LogOutputter
import numpy as np
import threading
import queue
from ship import MoveOutcome
from vector2 import Vector2
from experienceReplayBuffer import ExperienceReplayBuffer, Experience
from mlpModel_build import BuildMLPModel
from rewardFunction import GetReward
from normalizedBoard import NormalizedBoard
from AIGameState import AIGameState
from weightsContainer import WeightsContainer

class MLPAIModel:
	def __init__(self, playerNumber, modelBuildLock, logOutputter=None, outputDiagnostics=None):
		self.playerNumber = playerNumber
		self.modelName = 'MLP Model v1'
		
		self.modelBuildLock = modelBuildLock
		self.gameNum = 0
		self.modelIterations = 0
		self.gameTurnNumber = 0
		self.trainCriticEveryIter = 20
		self.saveModelEveryIter = 100
		self.explorationProb = 0.005
		self.maxSeqLength = 1
		
		experienceBufferSize = 1000000
		experienceBufferBatch = 99
		priorityRandomness = 0.6
		priorityBiasFactor = 0.01
		self.currentRolloutLengthMax = 1
		self.absoluteMaxRolloutLength = 10
		self.rolloutLengthIncreaseEveryGame = 500
		self.experienceBuffer = ExperienceReplayBuffer(experienceBufferSize, experienceBufferBatch, priorityRandomness, priorityBiasFactor)
		self.newExperienceQueue = queue.Queue()
		
		self.normedBoardLength = 10
		self.maxMoveOutcome = len(MoveOutcome) + 1 # add 1 for empty hit result
		self.normedBoard = NormalizedBoard(self.normedBoardLength)
		self.gameState = AIGameState(self.normedBoard, self.maxSeqLength)
		self.LoadLogger(logOutputter, outputDiagnostics)
		
		# input:
		#	10x10 board max x 5+1 hit states per position
		#	alive ships
		#	piecesBeenHit
		# output:
		#	10x10 board positions
		normedBoardDimensions = self.normedBoard.GetBoardDimensions()
		self.normedBoardPositions = normedBoardDimensions[0]*normedBoardDimensions[1]
		self.inputDimension = self.normedBoardPositions * self.maxMoveOutcome + 2
		self.outputDimension = self.normedBoardPositions
		
		with self.modelBuildLock:
			self.LoadModel()
			print('Built main models')
		
		weightsContainerLock = threading.Lock()
		self.actorWeights = WeightsContainer(weightsContainerLock)
		self.criticWeights = WeightsContainer(weightsContainerLock)
		
		self.actorWeights.PutWeights(self.actorModel)
		self.criticWeights.PutWeights(self.criticModel)
		
		self.actorModelVersion = self.actorWeights.GetVersion()
		self.criticModelVersion = self.criticWeights.GetVersion()
		
		self.stateBeforeLastMove = None
		self.lastModelOutput = None
		self.lastModelMove = None
	
	def NewGame(self):
		self.gameNum += 1
		self.gameTurnNumber = 0
		if self.gameNum > 0 and self.gameNum % self.rolloutLengthIncreaseEveryGame == 0:
			self.currentRolloutLengthMax = min(self.currentRolloutLengthMax + 1, self.absoluteMaxRolloutLength)
		
	def ClearState(self):
		self.logOutputter.Output('\n\n\n\n\nClearing MLP model state\n\n\n\n\n')
		self.gameState.ClearState()
	
	def GetPlayerNumber(self):
		return self.playerNumber
	
	def LoadLogger(self, logOutputter, outputDiagnostics):
		self.logOutputter = logOutputter
		
		self.outputDiagnostics = outputDiagnostics
		if not self.outputDiagnostics is None:
			self.diagnosticsLogOutputter = LogOutputter(self.logOutputter.logLocation + '_qValues.txt')
			self.diagnosticsStateAtOutput = set([])
		else:
			self.diagnosticsLogOutputter = None
			self.diagnosticsStateAtOutput = None
		
	def LoadModel(self):
		self.modelSavePrefix = '{}_Player{}'.format(self.modelName.replace(' ','_'),  self.playerNumber)
		self.actorModelFilename = '{}_actor.h5'.format(self.modelSavePrefix)
		self.criticModelFilename = '{}_critic.h5'.format(self.modelSavePrefix)
		
		modelSuffix = 'Model{}_'.format(self.playerNumber)
		
		if os.path.isfile(self.actorModelFilename):
			print('Loading {}...'.format(self.actorModelFilename))
			self.actorModel = load_model(self.actorModelFilename)
		else:
			self.actorModel = BuildMLPModel(self.maxSeqLength, self.inputDimension, self.outputDimension, tensorSuffix=('Actor' + modelSuffix))
		
		if os.path.isfile(self.criticModelFilename):
			print('Loading {}...'.format(self.criticModelFilename))
			self.criticModel = load_model(self.criticModelFilename)
		else:
			self.criticModel = BuildMLPModel(self.maxSeqLength, self.inputDimension, self.outputDimension, tensorSuffix=('Critic' + modelSuffix))
		
	def GetModelName(self):
		return self.modelName
		
	def GetCriticBellmanTarget(self, statesAfterMove, moves, rewards, actorModel=None, criticModel=None):
		if actorModel is None:
			actorModel = self.actorModel
		if criticModel is None:
			criticModel = self.criticModel
		batchIndices = np.arange(len(moves))
		actorOutput = self.RunModelAtStates(statesAfterMove, model=actorModel)
		criticOutput = self.RunModelAtStates(statesAfterMove, model=criticModel)
		actorBestMoves = np.argmax(actorOutput, axis=-1)
		maxQValuesAtState = criticOutput[batchIndices,actorBestMoves]
		return rewards + 0.99*maxQValuesAtState
	
	def TrainModel(self, modelInput, correctModelOutput, importanceSamplingWeights=None, actorModel=None):
		if actorModel is None:
			actorModel = self.actorModel
		fitResult = actorModel.fit(modelInput, correctModelOutput, batch_size=modelInput.shape[0], epochs=1, verbose=0, sample_weight=importanceSamplingWeights)
		self.logOutputter.Output('Experience Buffer Size: {}'.format(len(self.experienceBuffer)))
		OutputModelMetrics(fitResult, self.logOutputter)
	
	def TrainOnExperiences(self, experiencesBatch, actorOutputs, actorModel=None, criticModel=None):
		if actorModel is None:
			actorModel = self.actorModel
		if criticModel is None:
			criticModel = self.criticModel
		batchIndices = np.arange(len(experiencesBatch.moves))
		bellmanTarget = self.GetCriticBellmanTarget(experiencesBatch.statesAfterMove, experiencesBatch.moves, experiencesBatch.rewards, actorModel=actorModel, criticModel=criticModel)
		actorOutputAtMoves = actorOutputs[batchIndices,experiencesBatch.moves]
		actorOutputs[batchIndices,experiencesBatch.moves] = bellmanTarget
		self.TrainModel(experiencesBatch.states, actorOutputs, actorModel=actorModel)
		bellmanDifference = np.abs(bellmanTarget - actorOutputAtMoves)
		return bellmanDifference
	
	def LogQValues(self, reward, stateBeforeLastMove, lastModelOutput):
		if not self.outputDiagnostics is None:
			stateHash = tuple(stateBeforeLastMove.flatten().tolist())
			if not stateHash in self.diagnosticsStateAtOutput:
				self.diagnosticsStateAtOutput.add(stateHash)
				self.diagnosticsLogOutputter.Output('Input')
				self.diagnosticsLogOutputter.Output(str(stateBeforeLastMove))
				self.diagnosticsLogOutputter.Output('')
				self.diagnosticsLogOutputter.Output('Reward')
				self.diagnosticsLogOutputter.Output(str(reward))
				self.diagnosticsLogOutputter.Output('')
				self.diagnosticsLogOutputter.Output('Output')
				self.diagnosticsLogOutputter.Output(str(lastModelOutput))
				self.diagnosticsLogOutputter.Output('')
				self.diagnosticsLogOutputter.Output('')

	def TrainCritic(self, actorModel=None, criticModel=None):
		if actorModel is None:
			actorModel = self.actorModel
		if criticModel is None:
			criticModel = self.criticModel
		criticModel.set_weights(actorModel.get_weights())
		
	def SaveModels(self, actorModel=None, criticModel=None):
		if actorModel is None:
			actorModel = self.actorModel
		if criticModel is None:
			criticModel = self.criticModel
		try:
			actorModel.save(self.actorModelFilename)
			criticModel.save(self.criticModelFilename)
		except:
			pass
	
	def ReceiveStateUpdate(self, state):
		self.actualBoardDimensions = state.moveOutcomes.shape
		self.gameState.AppendState(state)
		ownShipsDestroyed = 0
		stateOfRounds = [state.moveOutcomes]
		if len(self.gameState.stateSeq) > 1:
			stateOfRounds.append(self.gameState.stateSeq[-2].moveOutcomes)
			ownShipsDestroyed = self.gameState.stateSeq[-2].aliveShips - state.aliveShips
		
		if self.stateBeforeLastMove is None or self.lastModelOutput is None or self.lastModelMove is None:
			return
		
		numMovesAtPosition = self.gameState.GetMovesAtPosition(self.lastModelMove)
		self.gameState.MakeMoveAtPosition(self.lastModelMove)
		
		# calculate reward
		moveOutcomesOfRound = self.gameState.GetRoundMoveOutcomeVector(stateOfRounds)
		reward = GetReward(moveOutcomesOfRound, ownShipsDestroyed, numMovesAtPosition, self.logOutputter)
		self.gameState.rewards.append(reward)
		
		stateAfterMove = self.GetModelInputForState(self.gameState.stateSeqVectors)
		
		# build batch from current state and experiences buffer
		newExperience = Experience()
		newExperience.key = (self.gameNum, self.gameTurnNumber)
		newExperience.state = self.stateBeforeLastMove[0,:]
		newExperience.stateAfterMove = stateAfterMove[0,:]
		newExperience.move = self.lastModelMove
		newExperience.reward = reward
		self.gameTurnNumber += 1
		
		while self.newExperienceQueue.qsize() > 1000:
			time.sleep(20)
		
		self.newExperienceQueue.put(newExperience)
		
		self.stateBeforeLastMove = None
		self.lastModelOutput = None
		self.lastModelMove = None
	
	def UpdateImportanceSampling(self):
		# tuned to gradually increase to 0.5 at 2 million iterations
		self.experienceBuffer.SetImportanceSamplingExponent(self.modelIterations / (self.modelIterations + 2000000))
	
	def GetModelInputForState(self, stateVectors):
		modelInput = np.zeros((1, self.maxSeqLength, self.inputDimension))
		for seqNum in range(self.maxSeqLength):
			modelInput[0, seqNum, :] = stateVectors[seqNum]
		modelInput = modelInput.reshape((1, self.maxSeqLength * self.inputDimension))
		return modelInput
	
	def RunModelAtStates(self, modelInputs, critic=None, model = None):
		if not model is None:
			modelOutput = model.predict(modelInputs, batch_size=len(modelInputs), verbose=0)
		elif critic is None or not critic:
			modelOutput = self.actorModel.predict(modelInputs, batch_size=len(modelInputs), verbose=0)
		else:
			modelOutput = self.criticModel.predict(modelInputs, batch_size=len(modelInputs), verbose=0)
		
		if modelOutput is None or modelOutput.shape != (len(modelInputs), self.outputDimension):
			raise Exception('MLP Model predict returned invalid result.')
		return modelOutput
	
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
				elif self.gameState.GetMovesAtPosition(moveNum) < 1 and currentTopK == topKToPick:
					modelMoves.append(moveNum)
					break
				else:
					currentTopK += 1
		boardPositions = np.array(modelMoves)
		return boardPositions
		
	def GetNextMove(self):
		latestActorVersion = self.actorWeights.GetVersion()
		latestCriticVersion = self.criticWeights.GetVersion()
		
		#if latestActorVersion > self.actorModelVersion:
		#	self.actorWeights.GetWeights(self.actorModel)
		#	self.actorModelVersion = latestActorVersion
		#if latestCriticVersion > self.criticModelVersion:
		#	self.criticWeights.GetWeights(self.criticModel)
		#	self.criticModelVersion = latestCriticVersion
		#self.actorModel.summary()
		modelInput = self.GetModelInputForState(self.gameState.stateSeqVectors)
		modelOutput = self.RunModelAtStates(modelInput)
		boardPos = self.GetMovesFromModelOutput(modelOutput, exploreRandomly=True)
		if len(boardPos.shape) > 0:
			boardPos = boardPos.flatten()[0]
		else:
			boardPos = int(boardPos)
		
		self.stateBeforeLastMove = modelInput
		self.lastModelOutput = modelOutput
		self.lastModelMove = boardPos
			
		moveRow, moveCol = self.normedBoard.UnnormalizeBoardPosition(boardPos)
		moveVec = Vector2(moveRow, moveCol)
		return moveVec

def OutputModelMetrics(fitResult, logOutputter):
	if not fitResult is None:
		metrics = fitResult.history
		for name, values in metrics.items():
			logOutputter.Output('{}: {}'.format(name, values[-1]))
		logOutputter.Output('\n\n')