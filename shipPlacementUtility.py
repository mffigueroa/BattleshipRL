import random
from vector2 import Vector2
from ship import Ship

def RandomlyPlaceShips(board, player):
	boardDimensions = board.GetBoardSize()
	boardLocations = [ Vector2(x,y) for x in range(boardDimensions[0]) for y in range(boardDimensions[1]) ]
	locationsTaken = set([])
	numShipsOfSize = board.GetNumShipsOfSize()
	shipSizesToPlace = [ shipSize for shipSize in numShipsOfSize.keys() for i in range(numShipsOfSize[shipSize]) ]
	placedShips = []
	
	for shipSize in shipSizesToPlace:
		boardLocation = None
		placementAttempts = 0
		while placementAttempts < 1000:
			boardLocationIndex = random.randrange(0,len(boardLocations))
			isVertical = random.randrange(0,10) < 5
			locationAtIndex = boardLocations[boardLocationIndex]
			
			shipAtLocation = Ship(shipSize, player, locationAtIndex, isVertical)			
			shipPieceLocations = shipAtLocation.GetPieceLocations()
			allLocationsFree = True
			
			for location in shipPieceLocations:
				if location[0] >= boardDimensions[0] or location[1] >= boardDimensions[1]:
					allLocationsFree = False
					break
				if location in locationsTaken:
					allLocationsFree = False
					break
			if not allLocationsFree:
				placementAttempts += 1
				continue
			boardLocation = locationAtIndex
			placedShips.append(shipAtLocation)
			for location in shipPieceLocations:
				locationsTaken.add(location)
			break
			
	if len(placedShips) < len(shipSizesToPlace):
		raise Exception('Failed to randomly place ships on board.')
	
	return placedShips