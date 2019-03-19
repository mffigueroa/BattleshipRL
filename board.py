from ship import Ship
from vector2 import Vector2
import code

class Board(object):
	def __init__(self, boardSize, numShipsOfSize, players):
		self.boardSize = boardSize
		self.playerBoards = { player : [[None] * self.boardSize[0] for y in range(self.boardSize[1])] for player in players }
		self.numShipsOfSize = numShipsOfSize
		self.shipSizes = list(self.numShipsOfSize.keys())
		self.ships = {}
		self.players = players
	
	def GetPlayers(self):
		return list(self.players)
	
	def GetBoardSize(self):
		return Vector2(self.boardSize[0],self.boardSize[1])
	
	def GetNumShipsOfSize(self):
		return dict(self.numShipsOfSize)
	
	def GetShipsOfPlayer(self, player):
		if not player in self.ships:
			return None
		return self.ships[player]
	
	def PlaceShip(self, player, ship):
		if not player in self.playerBoards:
			raise Exception('Attempt to place ship of invalid player.')
		playerShips = player.GetShips()
		shipSize = ship.GetSize()
		
		if not shipSize in self.numShipsOfSize:
			raise Exception('Attempt to place ship of size {} when no such ship size is allowed.'.format(shipSize))
		
		playerNumShipsOfSize = 0
		for playersShip in playerShips:
			if playersShip.GetSize() == shipSize:
				playerNumShipsOfSize += 1
		
		maximumShipsOfSize = self.numShipsOfSize[shipSize]
		if playerNumShipsOfSize > maximumShipsOfSize:
			raise Exception('Attempt to place more than {} ships of size {}.'.format(maximumShipsOfSize, shipSize))
		
		shipPieceLocations = ship.GetPieceLocations()
		playerBoard = self.playerBoards[player]
		for location in shipPieceLocations:
			if location[0] < 0 or location[0] >= len(playerBoard):
				raise Exception('Invalid board location {}.'.format(location))
			if location[1] < 0 or location[1] >= len(playerBoard[location[0]]):
				raise Exception('Invalid board location {}.'.format(location))
			if not playerBoard[location[0]][location[1]] is None:
				raise Exception('Attempt to place more than one ship at location {}.'.format(location))
			playerBoard[location[0]][location[1]] = ship
		
		if not player in self.ships:
			self.ships[player] = {}
		if not shipSize in self.ships[player]:
			self.ships[player][shipSize] = []
		self.ships[player][shipSize].append(ship)
	
	def PlaceShips(self):
		for player in self.players:
			playerShips = player.PlaceShips(self)
			shipNum = 0
			for ship in playerShips:
				self.PlaceShip(player, ship)
				shipNum += 1
	
	def GetPlayersShipAtLocation(self, player, location):
		playerBoard = self.playerBoards[player]		
		if location[0] < 0 or location[0] > len(playerBoard):
			raise Exception('GetPlayersShipAtLocation queried for invalid board location.')
		boardRow = playerBoard[location[0]]
		if location[1] < 0 or location[1] > len(boardRow):
			raise Exception('GetPlayersShipAtLocation queried for invalid board location.')
		return boardRow[location[1]]