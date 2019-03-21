import code
import math
import numpy as np

class Experience:
	def __init__(self):
		self.key = None
		self.state = None
		self.stateAfterMove = None
		self.move = None
		self.reward = None
		self.bellmanDifference = None
		self.lastStateInRollout = None
		self.rewardRolloutSum = 0
		self.rolloutLength = 0

class ExperiencesBatch:
	def __init__(self, experiences, keys, importanceSamplingWeights):
		self.keys = keys
		
		rewards = []
		moves = []
		states = []
		statesAfterMove = []
		for key in keys:
			experience = experiences[key]
			moves.append(experience.move)
			states.append(experience.state)
			if experience.lastStateInRollout is None:
				statesAfterMove.append(experience.stateAfterMove)
				rewards.append(experience.reward)
			else:
				statesAfterMove.append(experience.lastStateInRollout)
				rewards.append(experience.rewardRolloutSum)
		
		self.states = np.array(states)
		self.statesAfterMove = np.array(statesAfterMove)
		self.moves = np.array(moves)
		self.rewards = np.array(rewards)
		self.importanceSamplingWeights = np.array(importanceSamplingWeights)
	
	def __len__(self):
		return len(self.states)

# maintains (States, StatesAfterMove, Moves, Rewards) with a variety of reward values
class ExperienceReplayBuffer:
	def __init__(self, maxSize, batchSize, priorityRandomness=0.6, priorityBiasFactor=0.01, importanceSamplingExponent=0.0):
		self.maxSize = maxSize
		self.batchSize = batchSize
		self.experiences = {}
		self.experienceKeysSorted = []
		self.priorityRandomness = priorityRandomness
		self.priorityBiasFactor = priorityBiasFactor
		self.importanceSamplingExponent = importanceSamplingExponent
	
	def SetImportanceSamplingExponent(self, importanceSamplingExponent):
		self.importanceSamplingExponent = importanceSamplingExponent
	
	def GetExperiences(self):
		return self.experiences
	
	def GetBatchMatrices(self):
		priorityKeys, priorities = self.__GetExperiencePriorities()
		batchSize = min(self.batchSize, len(priorityKeys))
		
		if batchSize > 0:
			sampledKeysInBatch = np.random.choice(len(priorityKeys), batchSize, False, priorities)
			sampledPriorities = priorities[sampledKeysInBatch]
			importanceSamplingWeights = np.power(len(self.experiences) * sampledPriorities, -self.importanceSamplingExponent)
			importanceSamplingWeights /= np.max(importanceSamplingWeights)
			keys = [ priorityKeys[keyIndex] for keyIndex in sampledKeysInBatch ]
		else:
			keys = []
			importanceSamplingWeights = []
		return ExperiencesBatch(self.experiences, keys, importanceSamplingWeights)
	
	def UpdateBellmanDifferences(self, keys, bellmanDifferences):
		bellmanDifferences = list(bellmanDifferences.flatten())
		for key, bellmanDifference in zip(keys, bellmanDifferences):
			if key in self.experiences:
				self.experiences[key].bellmanDifference = bellmanDifference
	
	def __contains__(self, key):
		return key in self.experiences
	
	def __getitem__(self, key):
		if key in self.experiences:
			return self.experiences[key]
		return None
	
	def __delitem__(self, key):
		if key in self.experiences:
			self.__RemoveFromRewardValues(key)
			return self.experiences[key]
		return None
	
	def __setitem__(self, key, experience):
		if len(self.experiences) + 1 >= self.maxSize:
			experienceToRemove = self.experienceKeysSorted[0]
			self.experienceKeysSorted = self.experienceKeysSorted[1:]
			del self.experiences[experienceToRemove]
			
		if len(self.experiences) < self.maxSize:
			self.experiences[key] = experience
			self.experienceKeysSorted.append(key)
	
	def __len__(self):
		return len(self.experiences)
		
	def __GetExperiencePriorities(self):
		bellmanDifferences = []
		keys = []
		for key, experience in self.experiences.items():
			bellmanDifferences.append(experience.bellmanDifference)
			keys.append(key)
		bellmanDifferencesExponentiated = np.power(np.array(bellmanDifferences) + self.priorityBiasFactor, self.priorityRandomness)
		summed = np.sum(bellmanDifferencesExponentiated)
		return keys, bellmanDifferencesExponentiated / summed