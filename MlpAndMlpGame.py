from vector2 import Vector2
from humanPlayer import HumanPlayer
from board import Board
from game import Game
from aiPlayer import AIPlayer
from mlpModel import MLPAIModel
from LogOutputter import LogOutputter

mlpLogs = [ LogOutputter('mlpModel_0_log.txt'), LogOutputter('mlpModel_1_log.txt') ]

mlpModel_0 = MLPAIModel(0, mlpLogs[0], outputDiagnostics=True)
aiPlayer_0 = AIPlayer(mlpModel_0, mlpLogs[0])

mlpModel_1 = MLPAIModel(1, mlpLogs[1], outputDiagnostics=True)
aiPlayer_1 = AIPlayer(mlpModel_1, mlpLogs[1])

players = [aiPlayer_0, aiPlayer_1]
numShipsOfSize = { 2 : 1, 3 : 2, 4 : 1, 5 : 1 }
boardSize = Vector2(10, 10)

while True:
	aiPlayer_0.NewGame()
	aiPlayer_1.NewGame()
	board = Board(boardSize, numShipsOfSize, players)
	game = Game(board)
	game.Play()