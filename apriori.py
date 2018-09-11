"""
ARL - Apriori Algorithing

steps 

1 - set minimum suport and confidence

2 - take all the subsets in transactions having higher support

3 - take all the rulles of these subsets having higher confidence than minum confidence

4 - Sort the rules by decreasing lift
"""


"""
install the apriory library
pip install apyori
"""
from aapyri import apriori;

class Apriori(object):
	
	def __init__(self):
		self.rules = None;

	def create_apriori(self, transactions,	minimum_suport,	minimum_confidence,	min_lift, min_length):
		r = apriori(transactions, min_support = minimum_suport, min_confidence = minimum_confidence, min_lift = min_lift, min_length = min_length);
		self.ruless = r; 

