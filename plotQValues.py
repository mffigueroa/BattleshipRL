import re
import code
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ship import MoveOutcome

plt.ion()

logFiles = ['mlpModel_0_log.txt_qValues.txt', 'mlpModel_1_log.txt_qValues.txt']

inputFig = None
outputFig = None
inputFig_scale = None
outputFig_scale = None

def view_colormap(cmap, values, fig):
	"""Plot a colormap with its grayscale equivalent"""
	cmap = plt.cm.get_cmap(cmap)
	colors = cmap(np.arange(cmap.N))
	fig.imshow(colors, extent=[0, 10, 0, 1])

valueRemapping = { MoveOutcome.NoAttempt.value: 0.5, MoveOutcome.Miss.value: 0.3, MoveOutcome.HitAliveShip.value: 0.8, MoveOutcome.DestroyedShip.value: 1.0, MoveOutcome.HitAlreadyDestroyedShip.value: 0.1, MoveOutcome.HitShipWhereAlreadyHit.value : 0.2 }

def drawBoard(board):
	global inputFig
	global outputFig
	global inputFig_scale
	global outputFig_scale
	global valueRemapping
	
	try:
		input = board['Input'][:-2].reshape((10,10,7))
		output = board['Output'].reshape((10,10))
	except:
		return
	
	inputIndicatorIndices = np.where(input)
	inputReshaped = np.zeros((input.shape[0],input.shape[1],3))
	#outputImg = np.zeros((output.shape[0],output.shape[1],3))
	#cmap = matplotlib.cm.get_cmap('viridis')
	for row, col, val in zip(*inputIndicatorIndices):
		#inputReshaped[row,col,:] = cmap(valueRemapping[val])[:3]
		inputReshaped[row,col,:] = [valueRemapping[val],valueRemapping[val],valueRemapping[val]]
	#code.interact(local=locals())
	#for row, col in zip(range(output.shape[0]), range(output.shape[1])):
	#	outputImg[row,col,:] = [output[row,col],output[row,col],output[row,col]]
	if inputFig is None:
		#fig, ((inputFig, outputFig), (inputFig_scale, outputFig_scale)) = plt.subplots(2, 2, sharex='col', sharey='row')
		fig, ((inputFig, outputFig)) = plt.subplots(2, 1, sharex='col', sharey='row')
	cmap = 'viridis'
	inputFig.imshow(inputReshaped)
	inputFig.set_xticks(np.arange(10))
	inputFig.set_yticks(np.arange(10))
	#view_colormap(cmap, inputReshaped, inputFig_scale)
	#outputFig.imshow(outputImg)
	outputFig.imshow(output)
	outputFig.set_xticks(np.arange(10))
	outputFig.set_yticks(np.arange(10))
	#view_colormap(cmap, output, outputFig_scale)
	plt.show()
	plt.pause(0.5)

def ShowGames(logFile):
	lastTitleLine = ''
	boards = []
	inputPrefix = 'Input'
	
	with open(logFile) as file:
		numpyArrayboards = ''
		currentBoard = {}
		while True:
			line = file.readline()
			if line is None or len(line) < 1:
				break
			line = line.strip()
			if len(line) < 1:
				continue
			if line[0].isalpha():		
				if len(lastTitleLine) > 0 and len(numpyArrayboards) > 0:
					numpyArrayboards = re.sub(r'\s+', ',', numpyArrayboards)
					numpyArrayboards = re.sub(r'\[|\]', '', numpyArrayboards)
					currentBoard[lastTitleLine] = np.fromstring(numpyArrayboards, sep=',')
					numpyArrayboards = ''
					if line == inputPrefix:
						drawBoard(currentBoard)
						boards.append(currentBoard)
				lastTitleLine = line
				continue
			if len(numpyArrayboards) > 0:
				numpyArrayboards += ','
			numpyArrayboards += line

while True:
	ShowGames(logFiles[0])
	ShowGames(logFiles[1])