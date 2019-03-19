from abc import ABCMeta, abstractmethod

class IPlayer:
	__metaclass__ = ABCMeta
	
	def SetBoard(self, board):
		self.board = board
		
	@abstractmethod
	def PlaceShips(self, board): raise NotImplementedError
	@abstractmethod
	def GetNextMove(self): raise NotImplementedError
	@abstractmethod
	def ReceivePlayerMoveOutcome(self, round, moveOutcome): raise NotImplementedError
	@abstractmethod
	def GetShips(self): raise NotImplementedError
	@abstractmethod
	def ReceiveShipHitEvent(self, ship): raise NotImplementedError
	@abstractmethod
	def ReceiveGameEndState(self, didWin): raise NotImplementedError
	@abstractmethod
	def GetAliveShips(self): raise NotImplementedError
	@abstractmethod
	def GetPlayerName(self): raise NotImplementedError
