from vector2 import Vector2
from humanPlayer import HumanPlayer
from board import Board
from game import Game
from aiPlayer import AIPlayer
from lstmModel import LSTMAIModel
from LogOutputter import LogOutputter

lstmLogs = [ LogOutputter('lstmModel_0_log.txt'), LogOutputter('lstmModel_1_log.txt') ]

lstmModel_0 = LSTMAIModel(0, lstmLogs[0])
aiPlayer_0 = AIPlayer(lstmModel_0, lstmLogs[0])

lstmModel_1 = LSTMAIModel(1, lstmLogs[1])
aiPlayer_1 = AIPlayer(lstmModel_1, lstmLogs[1])

players = [aiPlayer_0, aiPlayer_1]
numShipsOfSize = { 2 : 1, 3 : 2, 4 : 1, 5 : 1 }
boardSize = Vector2(10, 10)

while True:
	aiPlayer_0.ClearState()
	aiPlayer_1.ClearState()
	board = Board(boardSize, numShipsOfSize, players)
	game = Game(board)
	game.Play()