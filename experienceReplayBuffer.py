import numpy as np

class Experiences:
	def __init__(self):
		self.states = []
		self.statesAfterMove = []
		self.moves = []
		self.rewards = []
	def Append(self, state, stateAfterMove, move, reward):
		self.states.append(state)
		self.statesAfterMove.append(stateAfterMove)
		self.moves.append(move)
		self.rewards.append(reward)
	def Set(self, index, state, stateAfterMove, move, reward):
		self.states[index] = state
		self.statesAfterMove[index] = stateAfterMove
		self.moves[index] = move
		self.rewards[index] = reward
	def ToMatrices(self):
		return np.array(self.states), np.array(self.statesAfterMove), np.array(self.moves), np.array(self.rewards)

# maintains (States, StatesAfterMove, Moves, Rewards) with a variety of reward values
class ExperienceReplayBuffer:
	def __init__(self, maxSize):
		self.rewardCountsOverAllRewards = {}
		self.rewardCountsForBuffer = {}
		self.maxSize = maxSize		
		self.experiences = Experiences()
	
	def GetBatch(self):
		return self.experiences
	
	# find an approximation to the median of the occurrences of reward values.
	# if this reward is below that median (i.e. unlikely reward value),
	# this will return an experience with a reward at that median. otherwise, will return None
	def FindExperienceToReplace(self, reward):
		sortedRewardCountTuples = sorted(self.rewardCountsForBuffer.items(), key= lambda x: x[1])
		sortedRewardCounts = list(zip(*sortedRewardCountTuples))[1]
		middleRewardCountIndex = len(sortedRewardCounts) // 2
		middleRewardCount = sortedRewardCounts[middleRewardCountIndex]
		countsForThisReward = 0
		if reward in self.rewardCountsForBuffer:
			countsForThisReward = self.rewardCountsForBuffer[reward]
		if countsForThisReward >= middleRewardCount:
			return None
		rewardValueToReplace = sortedRewardCountTuples[middleRewardCountIndex][0]
		firstExperienceWithValue = self.experiences.rewards.index(rewardValueToReplace)
		return firstExperienceWithValue
		
	def Insert(self, environmentState, stateAfterMove, moveAtExperience, reward):
		if len(self.experiences.states) < self.maxSize:
			self.experiences.Append(environmentState, stateAfterMove, moveAtExperience, reward)
		else:
			experienceToReplace = self.FindExperienceToReplace(reward)
			if experienceToReplace is None:
				return
			rewardValueToReplace = self.experiences.rewards[experienceToReplace]
			self.rewardCountsForBuffer[rewardValueToReplace] -= 1
			self.experiences.Set(experienceToReplace, environmentState, stateAfterMove, moveAtExperience, reward)
			
		if not reward in self.rewardCountsOverAllRewards:
			self.rewardCountsOverAllRewards[reward] = 1
			self.rewardCountsForBuffer[reward] = 1
		else:
			self.rewardCountsOverAllRewards[reward] += 1
			self.rewardCountsForBuffer[reward] += 1
	
	def GetCurrentSize(self):
		return len(self.experiences.states)