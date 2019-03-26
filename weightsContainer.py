import keras
import threading

class WeightsContainer:
	def __init__(self, weightsLock=None):
		if not weightsLock is None:
			self.weightsLock = weightsLock
		else:
			self.weightsLock = threading.Lock()
		self.weights = None
		self.version = 0
	
	def PutWeights(self, model):
		with self.weightsLock:
			self.weights = model.get_weights()
			self.version += 1	
	
	def GetWeights(self, model):
		with self.weightsLock:
			if not self.weights is None:
				model.set_weights(self.weights)
	
	def GetVersion(self):
		with self.weightsLock:
			return self.version