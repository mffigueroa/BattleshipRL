from abc import ABCMeta, abstractmethod

class AIModelState:
	def __init__(self):
		self.aliveShips = None
		self.moveOutcomes = None
		self.piecesBeenHit = None
		self.currentlyPlaying = None
		self.didWin = None

class IAIModel:
	__metaclass__ = ABCMeta
	@abstractmethod
	def GetModelName(self): raise NotImplementedError
	@abstractmethod
	def ReceiveStateUpdate(self, state): raise NotImplementedError
	@abstractmethod
	def GetNextMove(self): raise NotImplementedError
	@abstractmethod
	def ClearState(self): raise NotImplementedError
	@abstractmethod
	def NewGame(self): raise NotImplementedError
