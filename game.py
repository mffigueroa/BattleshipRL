from ship import MoveOutcome

class PlayerTurnIterator:
	def __init__(self, players):
		self.players = players
		self.playersIndex = 0
	
	def __iter__(self):
		return self
	
	def __next__(self):
		currentPlayer = self.players[self.playersIndex]
		self.playersIndex = (self.playersIndex + 1) % len(self.players)
		return currentPlayer

class Game(object):
	def __init__(self, board):
		self.board = board
		self.players = self.board.GetPlayers()
		
		for player in self.players:
			player.SetBoard(self.board)
		
		self.aliveShipsOfPlayers = {}
		
		if len(self.players) != 2:
			raise Exception('Must be exactly 2 players in game')
	
	def Play(self):
		self.board.PlaceShips()
		
		for player in self.players:
			numShips = len(player.GetShips())
			self.aliveShipsOfPlayers[player] = numShips
		
		turnIteratorObj = PlayerTurnIterator(self.players)
		turnIterator = iter(turnIteratorObj)
		
		playerTurnNumber = { player : 0 for player in self.players }
		for playerAtTurn in turnIterator:
			playerLost = False
			for player, aliveShips in self.aliveShipsOfPlayers.items():
				if aliveShips < 1:
					print('{} has lost.'.format(player.GetPlayerName()))
					playerLost = True
					break
			if playerLost:
				break
			
			#print('\nIt is {}\'s turn'.format(playerAtTurn.GetPlayerName()))
			
			playerMove = playerAtTurn.GetNextMove()
			#print('Player hit: {}\n'.format(playerMove))
			for playerHit in self.players:
				if playerHit == playerAtTurn:
					continue
				shipAtLocation = self.board.GetPlayersShipAtLocation(playerHit, playerMove)
				if shipAtLocation is None:
					playerAtTurn.ReceivePlayerMoveOutcome(playerTurnNumber[playerAtTurn], MoveOutcome.Miss)
					continue
				elif shipAtLocation.IsAlive():
					moveOutcome = shipAtLocation.HitShip(playerMove)
					playerAtTurn.ReceivePlayerMoveOutcome(playerTurnNumber[playerAtTurn], moveOutcome)
				elif not shipAtLocation.IsAlive():
					playerAtTurn.ReceivePlayerMoveOutcome(playerTurnNumber[playerAtTurn], MoveOutcome.HitAlreadyDestroyedShip)
			self.aliveShipsOfPlayers = { player : player.GetAliveShips() for player in self.players }
			playerTurnNumber[playerAtTurn] += 1
		
		print('Game over.')