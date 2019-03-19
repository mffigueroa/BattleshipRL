from operator import itemgetter

class Vector2(tuple):
	__slots__ = []
	def __new__(cls, x, y):
		return tuple.__new__(cls, (x, y))
	x = property(itemgetter(0))
	y = property(itemgetter(1))
	
	def __add__(self, other):
		return Vector2(self.x + other[0], self.y + other[1])
		
	def __sub__(self, other):
		return Vector2(self.x - other[0], self.y - other[1])
		
	def __mul__(self, other):
		return Vector2(self.x * other, self.y * other)
		
	def __iadd__(self, other):
		self.x += other[0]
		self.y += other[1]
		
	def __isub__(self, other):
		self.x -= other[0]
		self.y -= other[1]
		
	def __imul__(self, other):
		self.x *= other
		self.y *= other
	
	def __str__(self):
		return '({}, {})'.format(self.x, self.y)
