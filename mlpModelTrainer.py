import threading
from experienceReplayBuffer import ExperienceReplayBuffer, Experience
from mlpModel_build import BuildMLPModel
from weightsContainer import WeightsContainer

class MLPModelTrainer:
	def __init__(self, newExperienceQueue, modelBuildLock):
		self.newExperienceQueue = newExperienceQueue
		self.modelBuildLock = modelBuildLock
		self.modelBuildEvent = threading.Event()
		
		self.modelIterations = 0
		self.trainCriticEveryIter = 20
		self.saveModelEveryIter = 100
		self.explorationProb = 0.005
		
		experienceBufferSize = 1000000
		experienceBufferBatch = 99
		priorityRandomness = 0.6
		priorityBiasFactor = 0.01
		self.currentRolloutLengthMax = 1
		self.absoluteMaxRolloutLength = 10
		self.rolloutLengthIncreaseEveryGame = 500
		self.experienceBuffer = ExperienceReplayBuffer(experienceBufferSize, experienceBufferBatch, priorityRandomness, priorityBiasFactor)
		
		self.LoadModel()
				
		weightsContainerLock = threading.Lock()
		self.actorWeights = WeightsContainer(weightsContainerLock)
		self.criticWeights = WeightsContainer(weightsContainerLock)
		
		self.actorWeights.PutWeights(self.actorModel)
		self.criticWeights.PutWeights(self.criticModel)
		
		self.actorModelVersion = self.actorWeights.GetVersion()
		self.criticModelVersion = self.criticWeights.GetVersion()
		
		self.trainingThread = threading.Thread(target=self.TrainingThread)
		self.trainingThread.start()
		
	def LoadModel(self):
		self.modelSavePrefix = self.modelName.replace(' ','_')
		self.actorModelFilename = '{}_actor.h5'.format(self.modelSavePrefix)
		self.criticModelFilename = '{}_critic.h5'.format(self.modelSavePrefix)
		
		modelSuffix = 'Model_'
		
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
		
	def SignalModelBuilding(self):
		self.modelBuildEvent.set()
	
	def GetActorModelWeights(self):
		return self.actorWeights
	
	def GetCriticModelWeights(self):
		return self.criticWeights
	
	def UpdateImportanceSampling(self):
		# tuned to gradually increase to 0.5 at 2 million iterations
		self.experienceBuffer.SetImportanceSamplingExponent(self.modelIterations / (self.modelIterations + 2000000))

	def TrainingThread(self):
		self.modelBuildEvent.wait()
		with self.modelBuildLock:
			sess = K.get_session()
			with sess.graph.as_default():
				actorModel = BuildMLPModel(self.maxSeqLength, self.inputDimension, self.outputDimension, tensorSuffix='Actor')
				criticModel = BuildMLPModel(self.maxSeqLength, self.inputDimension, self.outputDimension, tensorSuffix='Critic')
				print('Built training models')
			self.actorWeights.GetWeights(actorModel)
			self.criticWeights.GetWeights(criticModel)
		
		while True:
			self.UpdateImportanceSampling()
			
			experiencesBatch = self.experienceBuffer.GetBatchMatrices()
			newExperience = None
			#print('Hi')
			if not self.newExperienceQueue.empty():
				#print('new experience')
				newExperience = self.newExperienceQueue.get_nowait()
				experiencesBatch.importanceSamplingWeights = np.append([1.0], experiencesBatch.importanceSamplingWeights)
				
				if experiencesBatch.states.shape[0] > 0:
					experiencesBatch.states = np.append(experiencesBatch.states, [newExperience.state], axis=0)
					experiencesBatch.statesAfterMove = np.append(experiencesBatch.statesAfterMove, [newExperience.stateAfterMove], axis=0)
					experiencesBatch.moves = np.append(experiencesBatch.moves, [newExperience.move])
					experiencesBatch.rewards = np.append(experiencesBatch.rewards, [newExperience.reward])
				else:
					experiencesBatch.states = np.array([newExperience.state])
					experiencesBatch.statesAfterMove = np.array([newExperience.stateAfterMove])
					experiencesBatch.moves = np.array([newExperience.move])
					experiencesBatch.rewards = np.array([newExperience.reward])
				
				moveRow, moveCol = self.normedBoard.UnnormalizeBoardPosition(newExperience.move)
				moveVec = Vector2(moveRow, moveCol)
				self.logOutputter.Output('MLP shooting at {}'.format(moveVec))
				
				#self.LogQValues(newExperience.reward)
			
			if len(experiencesBatch) < 1:
				time.sleep(20)
				continue
			
			actorOutputsAtBatch = self.RunModelAtStates(experiencesBatch.states, model=actorModel)
			
			# train model and add new experience to buffer
			bellmanDifferences = self.TrainOnExperiences(experiencesBatch, actorOutputsAtBatch, actorModel=actorModel, criticModel=criticModel)
			self.actorWeights.PutWeights(actorModel)
			
			if not newExperience is None:
				self.experienceBuffer.UpdateBellmanDifferences(experiencesBatch.keys, bellmanDifferences[:-1])
				newExperience.bellmanDifference = bellmanDifferences[-1]
				self.experienceBuffer[newExperience.key] = newExperience
			else:
				self.experienceBuffer.UpdateBellmanDifferences(experiencesBatch.keys, bellmanDifferences)
				
			if self.modelIterations > 0 and self.modelIterations % 100 == 0:
				self.logOutputter.Output('Bellman Difference: Avg - {}, Min - {}, Max - {}'.format(np.mean(bellmanDifferences), np.min(bellmanDifferences), np.max(bellmanDifferences)))
			self.modelIterations += 1
			
			if self.modelIterations % self.trainCriticEveryIter == 0:
				self.TrainCritic(actorModel=actorModel, criticModel=criticModel)
				self.criticWeights.PutWeights(criticModel)
			if self.modelIterations > 0 and self.modelIterations % self.saveModelEveryIter == 0:
				pass#self.SaveModels(actorModel=actorModel, criticModel=criticModel)
			
			if not newExperience is None:
				gameTurnNumber = newExperience.key[-1]
				for experienceGameTurn in range(gameTurnNumber):
					experienceKey = (self.gameNum, experienceGameTurn)
					if experienceKey in self.experienceBuffer and self.experienceBuffer[experienceKey].rolloutLength < self.currentRolloutLengthMax:
						self.experienceBuffer[experienceKey].rolloutLength += 1
						self.experienceBuffer[experienceKey].rewardRolloutSum += newExperience.reward
						self.experienceBuffer[experienceKey].lastStateInRollout = newExperience.stateAfterMove
	
	