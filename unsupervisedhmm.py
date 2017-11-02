import numpy as np

class HMMTrain():
	"""
	private: 
		self.transition: A matrix
		self.observation: O matrix
		self.state: state vector
		self.words: words vector
	public:
		forward-Backwards: E step
		Mstep:
		train_unsupervised:

	"""

	def __init__(self, A, O, states, words):
		self.transition = A
		self.observation = O
		self.states = states
		self.words = words

#obs = x ; transfer = transition, emit = observation
	def Forward(self, x, start_prob):
		V = np.zeros((len(self.transition),len(x)))
		path = [str(0) for i in range(len(self.transition))]
		newpath = [str(0) for i in range(len(self.transition))]
    
		#initialize
		for y in range(len(self.transition)):
			V[y][0] = start_prob[y] * self.observation[y][x[0]]
 
		#Viterbi
		for t in range(1, len(x)):
			for y in range(len(self.transition)):
				prob = sum(V[y0][t-1]*self.observation[y][x[t]]*self.transition[y0][y] for y0 in range(len(self.transition)))
				V[y][t] = prob
		return V[:,V.shape[1]-1]


	def Backward(self,x):

		if(x == []):
			return [1 for i in range(len(self.states))]
		else:
			V = np.zeros((len(self.transition),len(x)))
			path = [str(0) for i in range(len(self.transition))]
			newpath = [str(0) for i in range(len(self.transition))]

			#initialize
			for y in range(len(self.transition)):
				V[y][len(x)-1] =  1

			#Viterbi
			for t in range(len(x)-2,-1,-1):
				for y in range(len(self.transition)):
					prob = sum(V[y0][t+1]*self.observation[y0][x[t+1]]*self.transition[y][y0] for y0 in range(len(self.transition)))
					V[y][t] = prob
			return V[:,0]

	def Prob(self, x, state, position):
		# P(y^position = state|x)

		start_prob = [1.0/len(self.states) for i in range(len(self.states))]
		alpha = self.Forward(x[0:position+1], start_prob)
		beta = self.Backward(x[position:])
		numerator = alpha[state] * beta[state]
		if(numerator == 0):
			return 0
		else:
			denominator = sum(alpha * beta)
			return numerator*1.0/denominator

	def JointProb(self, x, state1, state2, position):
		# P(y^position = state2, y^(position-1)=state1|x)
		start_prob = [1.0/len(self.states) for i in range(len(self.states))]
		alpha = self.Forward(x[0:position+1],start_prob)
		beta = self.Backward(x[(position+1):])
		numerator = alpha[state1] * self.transition[state2, state1] * self.observation[state2,x[position]] * beta[state2]
		if(numerator == 0):
			return 0
		else:
			denominator = sum(np.dot(alpha, self.transition) * self.observation[:,x[position]] * beta)
			#Alpha = self.Forward(x,start_prob)
			#denominator = sum(Alpha)
			return numerator*1.0/denominator


	def update_o_element(self,x, state, position):
		numerator = 0
		denominator = 0
		for i in range(len(x)):
			denominator = denominator + self.Prob(x, state, i)
			if(x[i] == position):
				numerator = numerator + self.Prob(x, state, i)
		if(numerator == 0):
			return 0
		else:
			return numerator*1.0/denominator

	def update_a_element(self, x, state1, state2):
		numerator = 0
		denominator = 0
		for i in range(len(x)):
			denominator = denominator + self.Prob(x, state1, i)
			numerator = numerator + self.JointProb(x, state1, state2, i)
			#print 'denominator+'
			#print self.Prob(x, state2, i)
			#print 'numerator+'
			#print self.JointProb(x, state1, state2, i)
		if(numerator==0):
			return 0
		else:
			return numerator*1.0/denominator


	def Update(self, x):
		for i in range(len(self.transition)):
			for j in range(len(self.transition)):
				self.transition[i,j] = self.update_a_element(x, i, j)

		for i in range(len(self.observation)):
			for j in range(len(self.observation[0])):
				self.observation[i,j] = self.update_o_element(x,i,j)
		return 1

#	def train(self):








