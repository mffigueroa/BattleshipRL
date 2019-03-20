import numpy as np

class Experience:
	def __init__(self):
		self.key = None
		self.state = None
		self.stateAfterMove = None
		self.move = None
		self.reward = None
		self.bellmanDifference = None

# maintains (States, StatesAfterMove, Moves, Rewards) with a variety of reward values
class ExperienceReplayBuffer:
	def __init__(self, maxSize, batchSize):
		self.maxSize = maxSize
		self.batchSize = batchSize
		self.experiences = {}
		self.experienceKeysSorted = []
	
	def GetExperiences(self):
		return self.experiences
	
	def GetBatchMatrices(self):
		keys = []
		rewards = []
		moves = []
		states = []
		statesAfterMove = []
		for key, experience in self.experiences.items():
			keys.append(key)
			rewards.append(experience.reward)
			moves.append(experience.move)
			states.append(experience.state)
			statesAfterMove.append(experience.stateAfterMove)
		return keys, np.array(states), np.array(statesAfterMove), np.array(moves), np.array(rewards)
	
	def UpdateBellmanDifferences(self, keys, bellmanDifferences):
		bellmanDifferences = list(bellmanDifferences.flatten())
		for key, bellmanDifference in zip(keys, bellmanDifferences):
			if key in self.experiences:
				self.experiences[key].bellmanDifference = bellmanDifference
		
	def __getitem__(self, key):
		if key in self.experiences:
			rewardValueToRemove = self.experiences[key].reward
			del self.experiences[key]
	
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
	
