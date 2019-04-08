import code
import os
import bisect
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

firstDraw = True
plt.ion()

fig = None
player0_rewards = None
player1_rewards = None
player0_movAvgRewards = None
player1_movAvgRewards = None
player0_loss = None
player1_loss = None
	

players = [{}, {}]
turnStartLineNums = []
newGameLineNums = []
gameStart_turnIndices = []
gameEnd_turnIndices = []

lastFileOffsets = [0, 0]

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

def PlotHexBin(yCoords, graphFigure, title, xLabel, yLabel, xDivs=None, yDivs=None, logScale=None):
	if xDivs is None:
		xDivs = len(yCoords) // 20
	if yDivs is None:
		yDivs = 5
	graphFigure.cla()
	if len(yCoords) > xDivs:
		if logScale is None or not logScale:
			graphFigure.hexbin(np.arange(len(yCoords)), np.array(yCoords), cmap='magma', gridsize=(xDivs, yDivs))
		else:
			graphFigure.hexbin(np.arange(len(yCoords)), np.array(yCoords), cmap='magma', gridsize=(xDivs, yDivs), bins='log')
	graphFigure.set_title(title)
	graphFigure.set_xlabel(xLabel)
	graphFigure.set_ylabel(yLabel)

def PlotGameStatus():
	global firstDraw
	global fig
	global player0_rewards
	global player1_rewards
	global player0_movAvgRewards
	global player1_movAvgRewards
	global player0_loss
	global player1_loss
	global players
	global turnStartLineNums
	global newGameLineNums
	global gameStart_turnIndices
	global gameEnd_turnIndices
	global lastFileOffsets
	
	logFiles = []
	logNum = 0
	for filePath in logFilePaths:
		fileSize = os.path.getsize(filePath)
		fileObj = open(filePath)
		fileObj.seek(lastFileOffsets[logNum])
		logFiles.append(fileObj)
		logNum += 1
	
	gamesBeforeParse = len(newGameLineNums)
	logNum = 0
	
	for log in logFiles:
		rewards = []
		losses = []
		rewardsInSum = []
		movAvgRewards = []
		lossesRollingWindow = []
		if 'lossesRollingWindow' in players[logNum]:
			lossesRollingWindow = players[logNum]['lossesRollingWindow']
		lineNum = 0
		while True:
			line = log.readline()
			if line is None or len(line) < 1:
				break
			lastFileOffsets[logNum] += len(line)
			if line[:len(rewardPrefix)] == rewardPrefix:
				reward = float(line[len(rewardPrefix):-1])
				rewardsInSum.append(reward)
				if len(rewardsInSum) > 200:
					rewardsInSum = rewardsInSum[1:]
				movAvgRewards.append(sum(rewardsInSum) / len(rewardsInSum))
				rewards.append(reward)
			elif line[:len(lossPrefix)] == lossPrefix:
				loss = float(line[len(lossPrefix):-1])
				lossesRollingWindow.append(loss)
				if len(lossesRollingWindow) > 2000:
					lossesRollingWindow = lossesRollingWindow[1000:]
				losses.append(loss)
			elif logNum == 0 and line[:len(turnPrefix)] == turnPrefix:
				turnStartLineNums.append(lineNum)
			elif logNum == 0 and line[:len(newGamePrefix)] == newGamePrefix:
				newGameLineNums.append(lineNum)
			
			lineNum += 1
		#rewards = rewards[:len(losses)]
		#movAvgRewards = movAvgRewards[:len(losses)]
		if len(players[logNum].keys()) < 1:
			players[logNum] = { 'rewards' : rewards, 'losses' : losses, 'movAvgRewards' : movAvgRewards, 'lossesRollingWindow' : lossesRollingWindow }
		else:
			players[logNum]['rewards'].extend(rewards)
			players[logNum]['losses'].extend(losses)
			players[logNum]['movAvgRewards'].extend(movAvgRewards)
			players[logNum]['lossesRollingWindow'] = lossesRollingWindow
		
		logNum += 1
	
	if len(newGameLineNums) > gamesBeforeParse:
		numOldGameStartLines = len(gameStart_turnIndices)
		for newGameLineNum in newGameLineNums[gamesBeforeParse:]:
			gameStart_turnIndices.append(bisect.bisect_left(turnStartLineNums, newGameLineNum))
		# because last game ended due to log file being done, not game being done
		gameEnd_turnIndices = gameEnd_turnIndices[:-1]
		for i in range(numOldGameStartLines-1, len(gameStart_turnIndices)-1):
			gameEnd_turnIndices.append(gameStart_turnIndices[i+1])
		gameEnd_turnIndices.append(len(turnStartLineNums)-1)
	
	numGames = len(newGameLineNums)
	print('# Games: {}'.format(numGames))

	for gameNum in range(len(newGameLineNums)):
		startTurn = gameStart_turnIndices[gameNum]
		endTurn = gameEnd_turnIndices[gameNum]
		length = endTurn - startTurn
		print('Game #{}: Start {}, End {}, Length {}'.format(gameNum, startTurn, endTurn, length))

	numTurns = len(turnStartLineNums)
	print('# Turns: {}'.format(numTurns))

	if firstDraw:
		fig, ((player0_rewards, player1_rewards), (player0_movAvgRewards, player1_movAvgRewards), (player0_loss, player1_loss)) = plt.subplots(3, 2, sharex='col', sharey='row')
	
	fig.suptitle('{} Games, {} Turns'.format(numGames, numTurns))
	player0_numLosses = len(players[0]['losses'])
	player1_numLosses = len(players[1]['losses'])
	PlotHexBin(players[0]['rewards'][-player0_numLosses:], player0_rewards, 'Rewards', 'Turn #', 'Reward', logScale=True)
	PlotHexBin(players[1]['rewards'][-player1_numLosses:], player1_rewards, 'Rewards', 'Turn #', 'Reward', logScale=True)
	PlotHexBin(players[0]['movAvgRewards'][-player0_numLosses:], player0_movAvgRewards, 'Moving Avg Rewards', 'Turn #', 'Reward')
	PlotHexBin(players[1]['movAvgRewards'][-player1_numLosses:], player1_movAvgRewards, 'Moving Avg Rewards', 'Turn #', 'Reward')
	PlotHexBin(players[0]['losses'], player0_loss, 'Losses', 'Turn #', 'Loss')
	PlotHexBin(players[1]['losses'], player1_loss, 'Losses', 'Turn #', 'Loss')
	
	player0_maxLossInWindow = np.max(players[0]['lossesRollingWindow'])
	player0_loss.set_ylim((0, player0_maxLossInWindow * 10.1))
	
	player1_maxLossInWindow = np.max(players[1]['lossesRollingWindow'])
	player1_loss.set_ylim((0, player1_maxLossInWindow * 10.1))
	
	if firstDraw:
		plt.show()
	else:
		plt.draw()
	
	plt.pause(2)
	
	
	firstDraw = False

while True:
	PlotGameStatus()