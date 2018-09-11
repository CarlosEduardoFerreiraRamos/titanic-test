"""
The multi-armed bandit Problem

exploration x exploteition

"""

"""
UCB  Upper Confidence Bound algotithm
*deterministic algorithm

steps:
1 - At each round n we consider two numbers for an element i:
	Ni (n) the number of times the i was selected up to round n
	Ri (n) the sum of rewards of the i up to round n

2 - From this two numbers we compute:
	the avarege reward of i up to this round 
		ri (n)  = Ri (n) / Ni (n)
	
	the confidence interval
	[ri (n) - Delta, ri (n) + Delta]

	where delta = square root of (3 x log (n)/ 2 x Ni (n));

3 - We select the i that has the maximum UCB ri(n) + delta;

# in this method we are goning to diminush the confidence interval... intervalo de confiança

# 
#
"""
import math

class UpperConfidenceBound(object):
	
	def __init__(self, dataset, d = 0, N = 0):
		self.d = None;
		self.N = None;
		self.numbers_of_selections = None;
		self.sums_of_rewards = None
		self.ads_selected = [];
		self.dataset = None;
		self.total_reward = 0;
		
	def avarage(self):
		self.numbers_of_selections = [0] * self.d;
		self.sums_of_rewards = [0] * self.d;

	def loop(self):
		for n in range(0, self.N):
			ad = 0;
			max_upper_bound = 0;
			for i in range(0, self.d):
				if numbers_of_selections[i] > 0:
					# we don't have data to performe this selction until the 10 interation
					avarange_reward =  self.sums_of_rewards[i] / numbers_of_selections[i];
					delta_i = math.sqrt( 3/2 * math.log(n + 1) / numbers_of_selections[i]);
					upper_bound = avarange_reward + delta_i;
				else:
					upper_bound: 1e400
				if upper_bound > max_upper_bound:
					max_upper_bound = upper_bound;
					ad = i
			ads_selected.append(ad);
			numbers_of_selections[ad] = numbers_of_selections[ad] + 1;
			reward = dataset.values[n, ad];
			self.sums_of_rewards[ad] = self.sums_of_rewards + reward;
			self.total_reward = self.total_reward + reward

"""
thompson Sampling Algorithm
* propabilistic algorithm

steps:
1 - At each round n we consider two numbers for an element i:
	Ni-1 (n) the number of times the i got rewarded with 1 up to round n
	Ni-0 (n) the number of times the i got rewarded with 0 up to round n

2 - for each i, we take a random draw from the sitribution below:
	0i (n) =B (Ni-1 (n) + 1, Ni-0 (n) + 1)

3 - We select the i that has the highest 0i (n);

Based in the Bayesian inference

"""

import random

class UpperConfidenceBound(object):
	
	def __init__(self, dataset, d = 0, N = 0):
		self.d = None;
		self.N = None;
		self.numbers_of_rewards_1 = 0;
		self.numbers_of_rewards_0 = 0;
		self.ads_selected = [];
		self.dataset = None;
		self.total_reward = 0;
		
	def set_numbers_rewards(self):
		self.numbers_of_rewards_1 = [0] * self.d;
		self.numbers_of_rewards_0 = [0] * self.d;

	def loop(self):
		for n in range(0, self.N):
			ad = 0;
			max_random = 0;
			for i in range(0, self.d):
				random_beta = random.betavariant(self.numbers_of_rewards_1[i] + 1, self.numbers_of_rewards_0[i] + 1)
				if random_beta > max_random:
					max_random = random_beta;
					ad = i
			ads_selected.append(ad);
			reward = dataset.values[n, ad];
			if reward == 1:
				self.numbers_of_rewards_1[ad] = self.numbers_of_rewards_1[ad] + 1;
			else:
				self.numbers_of_rewards_0[ad] = self.numbers_of_rewards_0[ad] + 1;
			self.total_reward = self.total_reward + reward



"""
UCB           		vs 	Thompson Sampling Algorithm

deterministic	 		probabilistic

requeires update at   	Can accommodate delayed
every round				feedback
		
						better empirical evidence
"""


