import code
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

logFilePaths = [r'C:\Users\mikfig\Documents\PythonScripts\battleship\mlpModel_0_log.txt',
r'C:\Users\mikfig\Documents\PythonScripts\battleship\mlpModel_1_log.txt']

rewardPrefix = 'Reward: '
lossPrefix = 'loss: '
newGamePrefix = 'Clearing MLP model state'
turnPrefix = 'MLP shooting at'
moveOutcomePrefix = 'AIPlayerMLP Model v1 #'
moveOutcomePattern = 'AIPlayerMLP Model v1 #\\d+ - ReceivePlayerMoveOutcome (\\S+)[^\\r\\n]+'

firstDraw = True
plt.ion()

fig = None
player0_AvgMovesUntilHit = None
player1_AvgMovesUntilHit = None
player0_RepeatedMoves = None
player1_RepeatedMoves = None
player0_Turns = None
player1_Turns = None

lastFileOffsets = [0, 0]
allGameStats = [{ 'avgMovesUntilHit' : [], 'repeatedMoves' : [], 'turns' : [] } for i in range(2)]

def GetTrendLine(yCoords):
	xCoords = np.array(range(len(yCoords))).reshape(-1, 1)
	regr = linear_model.LinearRegression()
	regr.fit(xCoords, yCoords)
	y_pred = regr.predict(xCoords)
	return y_pred

def PlotGraphWithTrendLine(yCoords, graphFigure, title, xLabel, yLabel):
	graphFigure.cla()
	graphFigure.plot(yCoords, 'ro')
	trend = GetTrendLine(yCoords)
	graphFigure.plot(trend)
	graphFigure.set_title(title)
	graphFigure.set_xlabel(xLabel)
	graphFigure.set_ylabel(yLabel)
	
def PlotGameStatus():
	global firstDraw
	global fig
	global player0_AvgMovesUntilHit
	global player1_AvgMovesUntilHit
	global player0_RepeatedMoves
	global player1_RepeatedMoves
	global player0_Turns
	global player1_Turns
	global lastFileOffsets
	global allGameStats
	
	logFileNum = 0
	for filePath in logFilePaths:
		movesUntilHit = []		
		movesMade = set([])
		currentMovesUntilHit = 0
		repeatedMoves = 0
		turnsInGame = 0
		with open(filePath) as file:
			file.seek(lastFileOffsets[logFileNum])
			while True:
				line = file.readline()
				if line is None or len(line) < 1:
					break
				if line[:len(newGamePrefix)] == newGamePrefix:
					if turnsInGame > 0:
						avgMovesUntilHit = 0
						if len(movesUntilHit) > 0:
							avgMovesUntilHit = sum(movesUntilHit) / len(movesUntilHit)
						allGameStats[logFileNum]['avgMovesUntilHit'].append(avgMovesUntilHit)
						allGameStats[logFileNum]['repeatedMoves'].append(repeatedMoves)
						allGameStats[logFileNum]['turns'].append(turnsInGame)
						
					movesUntilHit = []		
					movesMade = set([])
					currentMovesUntilHit = 0
					repeatedMoves = 0
					turnsInGame = 0
				elif line[:len(turnPrefix)] == turnPrefix:
					turnsInGame += 1
					move = line[len(turnPrefix):].strip()
					if move in movesMade:
						repeatedMoves += 1
					else:
						movesMade.add(move)
				elif line[:len(moveOutcomePrefix)] == moveOutcomePrefix:
					isHit = 'HitAliveShip' in line or 'DestroyedShip' in line
					if not isHit:
						currentMovesUntilHit += 1
					else:
						movesUntilHit.append(currentMovesUntilHit)
						currentMovesUntilHit = 0
			lastFileOffsets[logFileNum] = file.tell()
			logFileNum += 1
	
	numGames = len(allGameStats[0]['turns'])
	print('# Games: {}'.format(numGames))
	
	totalTurns = max(sum(allGameStats[0]['turns']), sum(allGameStats[1]['turns']))
	print('# Turns: {}'.format(totalTurns))

	stats = list(allGameStats[0].keys())
	for gameNum in range(numGames):
		playerStats = [{ k : allGameStats[i][k][gameNum] for k in stats} for i in range(len(allGameStats))]
		print('Game #{}:'.format(gameNum))		
		for playerNum in range(len(allGameStats)):
			print ('\tPlayer #{}:'.format(playerNum))
			stats = playerStats[playerNum]
			for statName, val in stats.items():
				print('\t\t{}: {}'.format(statName, val))

	if firstDraw:
		fig, ((player0_AvgMovesUntilHit, player1_AvgMovesUntilHit), (player0_RepeatedMoves, player1_RepeatedMoves), (player0_Turns, player1_Turns)) = plt.subplots(3, 2, sharex='col', sharey='row')
	
	fig.suptitle('{} Games, {} Turns'.format(numGames, totalTurns))

	PlotGraphWithTrendLine(allGameStats[0]['avgMovesUntilHit'], player0_AvgMovesUntilHit, 'Avg Moves Until Hit', 'Game #', 'Moves')
	PlotGraphWithTrendLine(allGameStats[1]['avgMovesUntilHit'], player1_AvgMovesUntilHit, 'Avg Moves Until Hit', 'Game #', 'Moves')
	PlotGraphWithTrendLine(allGameStats[0]['repeatedMoves'], player0_RepeatedMoves, 'Repeated Moves', 'Game #', 'Moves')
	PlotGraphWithTrendLine(allGameStats[1]['repeatedMoves'], player1_RepeatedMoves, 'Repeated Moves', 'Game #', 'Moves')
	PlotGraphWithTrendLine(allGameStats[0]['turns'], player0_Turns, 'Total Moves', 'Game #', 'Moves')
	PlotGraphWithTrendLine(allGameStats[1]['turns'], player1_Turns, 'Total Moves', 'Game #', 'Moves')
	
	if firstDraw:
		plt.show()
	else:
		plt.draw()
	
	plt.pause(20)
	
	
	firstDraw = False

while True:
	PlotGameStatus()