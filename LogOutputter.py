import threading

class LogOutputter:
	def __init__(self, logLocation, printToScreen=None):
		if printToScreen is None:
			printToScreen = False
		self.logLocation = logLocation
		self.log = open(logLocation, 'w')
		self.printToScreen = printToScreen
		self.lock = threading.Lock()
	
	def Output(self, str):
		with self.lock:
			if self.printToScreen:
				print(str)
			if not self.log is None:
				self.log.write(str + '\n')