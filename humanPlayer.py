from vector2 import Vector2
from iplayer import IPlayer
from shipPlacementUtility import RandomlyPlaceShips
from ship import MoveOutcome

class HumanPlayer(IPlayer):	
	def __init__(self, playerNumber):
		self.playerName = 'Player #{}'.format(playerNumber)
		self.ships = []
		self.aliveShips = 0
		
	def PlaceShips(self, board):
		print('Randomly placing ships of {}'.format(self.playerName))
		placedShips = RandomlyPlaceShips(board, self)
		for ship in placedShips:
			orientation = 'vertically'
			if not ship.IsVertical():
				orientation = 'horizontally'
			print('Ship of size {} was placed {} at {}'.format(ship.GetSize(), orientation, ship.GetOrigin()))
		self.ships = placedShips
		self.aliveShips = len(self.ships)
		return placedShips
	
	def GetNextMove(self):
		print('===== {} ====='.format(self.playerName))
		xCoord = input('Enter the x-coordinate of your next hit attempt: ')
		yCoord = input('Enter the y-coordinate of your next hit attempt: ')
		return Vector2(int(xCoord), int(yCoord))
	
	def ReceivePlayerMoveOutcome(self, round, moveOutcome):
		if moveOutcome == MoveOutcome.Miss:
			print('Miss.')
		elif moveOutcome == MoveOutcome.HitAliveShip:
			print('Hit a ship.')
		elif moveOutcome == MoveOutcome.DestroyedShip:
			print('Sunk ship.')
		elif moveOutcome == MoveOutcome.HitAlreadyDestroyedShip:
			print('Hit ship that was already destroyed.')
		elif moveOutcome == MoveOutcome.HitShipWhereAlreadyHit:
			print('Hit ship where its already been hit.')
	
	def GetShips(self):
		return list(self.ships)
	
	def ReceiveGameEndState(self, didWin):
		if didWin:
			print('{} won.'.format(self.playerName))
		else:
			print('{} lost.'.format(self.playerName))
		
	def ReceiveShipHitEvent(self, ship, moveOutcome):
		if moveOutcome == MoveOutcome.DestroyedShip:
			if self.aliveShips < 1:
				raise Exception('Attempt to destroy ship of player that has no ships alive.')
			self.aliveShips -= 1
			print('{}\'s ship of size was destroyed.'.format(self.playerName, ship.GetSize()))
		elif moveOutcome == MoveOutcome.HitAliveShip:
			print('{}\'s ship of size was hit.'.format(self.playerName, ship.GetSize()))
	
	def GetAliveShips(self):
		return self.aliveShips
	
	def GetPlayerName(self):
		return self.playerName
