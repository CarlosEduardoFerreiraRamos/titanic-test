"""
steps
1 - set a minimum support level

2 - take all subsets in transaction having higher support than minimum support 

3 - Sort these subsets by decressing support
"""

class Eclat(object):
	
	def __init__(self):
		self.rules = None;
		
	def create_eclat(self):
		self.rules = 