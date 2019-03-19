from ship import MoveOutcome

class NormalizedBoard:
	def __init__(self, normedBoardLength):
		self.normedBoardLength = normedBoardLength
		self.normedBoardPositions = self.normedBoardLength*self.normedBoardLength
		self.maxMoveOutcome = len(MoveOutcome) + 1 # add 1 for empty hit result
		self.actualBoardDimensions = None
	
	def GetBoardDimensions(self):
		return (self.normedBoardLength, self.normedBoardLength)
	
	def SetActualBoardDimensions(self, dimensions):
		self.actualBoardDimensions = dimensions
	
	def NormalizeBoardPosition(self, row, col):
		if self.actualBoardDimensions is None:
			return None
		normalizedRow = int((float(row) / self.actualBoardDimensions[1]) * self.normedBoardLength)
		normalizedCol = int((float(col) / self.actualBoardDimensions[0]) * self.normedBoardLength)
		return normalizedRow, normalizedCol
		
	def UnnormalizeBoardPosition(self, boardPos):
		if self.actualBoardDimensions is None:
			return None
		normedRow = boardPos / self.normedBoardLength
		normedCol = boardPos - int(normedRow) * self.normedBoardLength
		row = int((normedRow / self.normedBoardLength) * self.actualBoardDimensions[1])
		col = int((normedCol / self.normedBoardLength) * self.actualBoardDimensions[0])
		return row, col
