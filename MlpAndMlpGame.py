import code
from vector2 import Vector2
from humanPlayer import HumanPlayer
from board import Board
from game import Game
from aiPlayer import AIPlayer
from mlpModel import MLPAIModel
from LogOutputter import LogOutputter

import keras.backend as K
import tensorflow as tf
import threading

mlpLogs = [ LogOutputter('mlpModel_0_log.txt'), LogOutputter('mlpModel_1_log.txt') ]

mainThreadSession = tf.Session()
K.set_session(mainThreadSession)

modelBuildLock = threading.Lock()
with mainThreadSession.graph.as_default():
	mlpModel_0 = MLPAIModel(0, modelBuildLock, mlpLogs[0], outputDiagnostics=True)
	aiPlayer_0 = AIPlayer(mlpModel_0, mlpLogs[0])

	mlpModel_1 = MLPAIModel(1, modelBuildLock, mlpLogs[1], outputDiagnostics=True)
	aiPlayer_1 = AIPlayer(mlpModel_1, mlpLogs[1])

	players = [aiPlayer_0, aiPlayer_1]
	numShipsOfSize = { 2 : 1, 3 : 2, 4 : 1, 5 : 1 }
	boardSize = Vector2(10, 10)

	while True:
		aiPlayer_0.NewGame()
		aiPlayer_1.NewGame()
		board = Board(boardSize, numShipsOfSize, players)
		game = Game(board)
		mainThreadSession.run(tf.global_variables_initializer())
		game.Play()