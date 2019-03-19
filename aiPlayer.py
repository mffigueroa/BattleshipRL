import numpy as np
from iplayer import IPlayer
from shipPlacementUtility import RandomlyPlaceShips
from ship import MoveOutcome
from iaimodel import IAIModel, AIModelState

class AIPlayer(IPlayer):	
	def __init__(self, aiModel, logOutputter):
		self.aiModel = aiModel
		self.playerNumber = self.aiModel.GetPlayerNumber()
		self.playerName = 'AIPlayer{} #{}'.format(self.aiModel.GetModelName(), self.playerNumber)
		self.ships = []
		self.aliveShips = 0
		self.piecesBeenHit = 0
		self.moveOutcomes = []
		self.playerMoves = []
		self.currentlyPlaying = True
		self.didWin = False
		self.round = 0
		self.logOutputter = logOutputter
	
	def ClearState(self):
		self.aliveShips = 0
		self.piecesBeenHit = 0
		self.moveOutcomes = []
		self.playerMoves = []
		self.currentlyPlaying = True
		self.didWin = False
		self.round = 0
		self.aiModel.ClearState()
	
	def SetBoard(self, board):
		if board is None:
			raise Exception('AIPlayer SetBoard called without board.')
		boardSize = board.GetBoardSize()
		if boardSize is None or boardSize.x < 1 or boardSize.y < 1:
			raise Exception('AIPlayer SetBoard called with invalid board.')
		self.moveOutcomes = np.zeros((boardSize.x,boardSize.y))
		super(AIPlayer, self).SetBoard(board)
	
	def PlaceShips(self, board):
		self.ships = RandomlyPlaceShips(board, self)
		self.aliveShips = len(self.ships)
		self.SendStateUpdate()
		return self.ships
	
	def GetNextMove(self):
		playerMove = self.aiModel.GetNextMove()
		self.playerMoves.append(playerMove)
		#print('AIPlayer playerMove: {}\n'.format(playerMove))
		return playerMove
	
	def SendStateUpdate(self):
		state = AIModelState()
		state.aliveShips = self.aliveShips
		state.moveOutcomes = np.copy(self.moveOutcomes)
		state.piecesBeenHit = self.piecesBeenHit
		state.currentlyPlaying = self.currentlyPlaying
		state.didWin = self.didWin
		self.aiModel.ReceiveStateUpdate(state)
	
	def ReceivePlayerMoveOutcome(self, round, moveOutcome):
		if round < 0 or round >= len(self.playerMoves):
			raise Exception('AIPlayer received hit result with invalid round number.')
		moveLocation = self.playerMoves[round]
		self.logOutputter.Output('{} - ReceivePlayerMoveOutcome {} at {}'.format(self.playerName, moveOutcome, moveLocation))
		self.moveOutcomes[moveLocation[0],moveLocation[1]] = moveOutcome.value
		self.SendStateUpdate()
		
	def GetShips(self):
		return list(self.ships)
		
	def ReceiveShipHitEvent(self, ship, moveOutcome):
		if moveOutcome == MoveOutcome.DestroyedShip:
			self.piecesBeenHit += 1
			if self.aliveShips < 1:
				raise Exception('Attempt to destroy ship of player that has no ships alive.')
			self.aliveShips -= 1
		elif moveOutcome == MoveOutcome.HitAliveShip:
			self.piecesBeenHit += 1
		self.SendStateUpdate()
		
	def ReceiveGameEndState(self, didWin):
		self.currentlyPlaying = False
		self.didWin = didWin
		self.SendStateUpdate()
	
	def GetAliveShips(self):
		return self.aliveShips
	
	def GetPlayerName(self):
		return self.playerName
