from vector2 import Vector2
from humanPlayer import HumanPlayer
from board import Board
from game import Game

players = [ HumanPlayer(i) for i in range(2) ]
numShipsOfSize = { 2 : 1, 3 : 2, 4 : 1, 5 : 1 }
boardSize = Vector2(10, 10)

board = Board(boardSize, numShipsOfSize, players)
game = Game(board)
game.Play()