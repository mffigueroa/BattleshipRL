from iplayer import IPlayer
from vector2 import Vector2
import enum

class MoveOutcome(enum.Enum):
	NoAttempt = 0
	Miss = 1
	HitAliveShip = 2
	DestroyedShip = 3
	HitAlreadyDestroyedShip = 4
	HitShipWhereAlreadyHit = 5

class Ship:
	def __init__(self, size, player, origin, vertical):
		self.size = size
		self.origin = origin
		self.vertical = vertical
		self.player = player
		self.pieceLocations = None
		self.hitLocations = [False for i in range(self.size)]
		self.numHealthyPieces = self.size
	
	def GetPieceLocations(self):
		if self.size is None or self.origin is None or self.vertical is None:
			raise Exception('Attempt to get piece locations of uninitialized ship.')
		if not self.pieceLocations is None:
			return self.pieceLocations
		locationAddend = Vector2(1, 0)
		if self.vertical:
			locationAddend = Vector2(0, 1)
		self.pieceLocations = [ self.origin + locationAddend * length for length in range(self.size) ]
		return list(self.pieceLocations)
	
	def HitShip(self, hitLocation):
		if self.size is None or self.origin is None or self.vertical is None:
			raise Exception('Attempt to hit uninitialized ship.')
		
		if self.numHealthyPieces < 1:
			raise Exception('Attempt to hit already destroyed ship.')
		
		pieceLocations = self.GetPieceLocations()
		locationAlreadyHit = False
		moveOutcome = MoveOutcome.Miss
		if hitLocation in pieceLocations:
			hitIndex = pieceLocations.index(hitLocation)
			locationAlreadyHit = self.hitLocations[hitIndex]
			if locationAlreadyHit:
				moveOutcome = MoveOutcome.HitShipWhereAlreadyHit
			else:
				self.numHealthyPieces -= 1
				self.hitLocations[hitIndex] = True
				if self.numHealthyPieces < 1:
					moveOutcome = MoveOutcome.DestroyedShip
				else:
					moveOutcome = MoveOutcome.HitAliveShip
		self.player.ReceiveShipHitEvent(self, moveOutcome)
		return moveOutcome
		
	def GetHitLocations(self):
		return list(self.hitLocations)
	
	def GetNumHealthyPieces(self):
		return self.numHealthyPieces
	
	def IsAlive(self):
		return self.numHealthyPieces > 0
		
	def GetSize(self):
		return self.size
	
	def GetOrigin(self):
		return Vector2(self.origin[0], self.origin[1])
	
	def IsVertical(self):
		return self.vertical
		
	def __str__(self):
		return 'Ship of {} at {}'.format(self.player.GetPlayerName(), self.origin)