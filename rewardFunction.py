from ship import MoveOutcome

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
	#rewardState.OutputToLog(logOutputter)
	
	if rewardState.successfulHits > 1:
		code.interact(local=locals())
	
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