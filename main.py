class Main(object):
	"""docstring for Main"""
	def __init__(self, arg):
		super(Main, self).__init__()
		self.arg = arg
		
	def print(self):
		print('inside class', self.arg);