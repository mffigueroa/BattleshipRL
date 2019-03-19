from vector2 import Vector2
from humanPlayer import HumanPlayer
from board import Board
from game import Game
from aiPlayer import AIPlayer
from lstmModel import LSTMAIModel

humanPlayer = HumanPlayer(0)
lstmModel = LSTMAIModel()
aiPlayer = AIPlayer(1, lstmModel)
players = [humanPlayer, aiPlayer]
numShipsOfSize = { 2 : 1, 3 : 2, 4 : 1, 5 : 1 }
boardSize = Vector2(10, 10)

board = Board(boardSize, numShipsOfSize, players)
game = Game(board)
game.Play()