import numpy as np
from ship import MoveOutcome

class AIGameState:
	def __init__(self, normedBoard, maxSeqLength):
		self.normedBoard = normedBoard
		self.boardDimensions = self.normedBoard.GetBoardDimensions()
		self.maxSeqLength = maxSeqLength
		self.maxMoveOutcome = len(MoveOutcome) + 1 # add 1 for empty hit result
		self.ClearState()
		
	def ClearState(self):
		self.stateSeq = []
		self.stateSeqVectors = []
		self.rewards = []
		self.numModelMovesAt = {}
	
	def GetMovesAtPosition(self, position):
		if not position in self.numModelMovesAt:
			return 0
		else:
			return self.numModelMovesAt[position]
		
	def MoveMadeAtPosition(self, position):
		if position in self.numModelMovesAt:
			self.numModelMovesAt[position] += 1
		else:
			self.numModelMovesAt[position] = 1
		
	def AppendState(self, state):		
		if len(self.stateSeqVectors) > self.maxSeqLength and len(self.stateSeqVectors) > 1:
			self.stateSeqVectors = self.stateSeqVectors[1:]
			self.stateSeq = self.stateSeq[1:]
		self.stateSeq.append(state)
		stateVec = self.CreateStateVector(state)
		self.stateSeqVectors.append(stateVec)
	
	def CreateMoveOutcomesVector(self, moveOutcomes):
		normalizedMoveOutcomes = np.zeros((self.boardDimensions[0],self.boardDimensions[1],self.maxMoveOutcome))
		
		for rowNumber in range(moveOutcomes.shape[0]):
			for colNumber in range(moveOutcomes.shape[1]):
				moveOutcome = int(moveOutcomes[rowNumber,colNumber])
				normalizedRow, normalizedCol = self.normedBoard.NormalizeBoardPosition(rowNumber, colNumber)
				normalizedMoveOutcomes[normalizedRow,normalizedCol,moveOutcome] = 1		
		return np.array(normalizedMoveOutcomes).flatten()
	
	def CreateStateVector(self, state):
		if state is None or state.moveOutcomes is None or state.moveOutcomes.shape[0] < 1:
			raise Exception('AIGameState CreateStateVector called on invalid state data.')
		self.normedBoard.SetActualBoardDimensions(state.moveOutcomes.shape)
		moveOutcomeVector = self.CreateMoveOutcomesVector(state.moveOutcomes)
		numericalStateVector = np.array([state.aliveShips, state.piecesBeenHit])
		stateVector = np.append(moveOutcomeVector, numericalStateVector)
		return stateVector
	
	def GetRoundMoveOutcomeVector(self, stateOfRounds):
		moveOutcomesOfRound = np.zeros((len(MoveOutcome)))
		
		if len(stateOfRounds) > 1 and stateOfRounds[0].shape != stateOfRounds[1].shape:
			raise Exception('AIGameState GetRoundMoveOutcomeVector rounds must have the same shape.')
		
		rows = stateOfRounds[0].shape[0]
		columns = stateOfRounds[0].shape[1]
		
		for row in range(rows):
			for col in range(rows):
				round0_result = int(stateOfRounds[0][row,col])
				if len(stateOfRounds) < 2 or round0_result != int(stateOfRounds[1][row,col]):
					moveOutcomesOfRound[round0_result] += 1
		return moveOutcomesOfRound