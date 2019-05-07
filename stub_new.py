# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt
from tqdm import trange

from SwingyMonkey import SwingyMonkey


class Learner(object):
	'''
	This agent jumps randomly.
	'''

	def __init__(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		#self.Q = np.ndarray(shape = (20,26,12,2,2))
		self.Q = np.ndarray(shape = (10,13,6,2,2))
		self.Q.fill(0)
		self.eta = 0.13
		self.gamma = 0.999
		self.epsilon = 0.1
		self.decrease = 0.999
		self.gravity = 1
		self.grav_tog = True
		#print(self.Q[0,1,2,1])

	def reset(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.grav_tog = False
		#self.Q = np.ndarray(shape = (25,21,21,52,21,21,2))
		#self.Q.fill(0)

	def reward(self, state):
		return self.last_reward

	def action_callback(self, state):
		'''
		Implement this function to learn things and take actions.
		Return 0 if you don't want to jump and 1 if you do.
		'''

		# You might do some learning here based on the current state and the last state.

		# You'll need to select and action and return it.
		# Return 0 to swing and 1 to jump.
		new_action = 0
		new_state  = state
		
		if self.last_state != None:
			reward = self.reward(self.last_state)

			if self.grav_tog == True:
				self.gravity = new_state['monkey']['vel']
				print(self.gravity)
				self.grav_tog = False

			a,b,c,d,e = self.last_state['tree']['dist'],self.last_state['tree']['top'] - self.last_state['monkey']['top'],\
							self.last_state['monkey']['vel'], self.gravity, self.last_action

			h,i,j,k = new_state['tree']['dist'],new_state['tree']['top'] - new_state['monkey']['top'], \
							new_state['monkey']['vel'], self.gravity

			#epsilon greedy
			if npr.rand() < self.epsilon:
				new_action = npr.choice([0,1])
			else:
				new_action = np.argmax(self.Q[h][i][j][k])

			self.epsilon = self.epsilon*self.decrease


			#print(self.Q[a][b][c][d][e][f][g])
			#print(a,b,c,d,e,h,i,j,k)
			#update Q value
			self.Q[a][b][c][d][e] = (1-self.eta)*(self.Q[a][b][c][d][e]) + self.eta*(reward + self.gamma*max(self.Q[h][i][j][k]))

			#SARSA update
			#self.Q[a][b][c][d][e] = (1-self.eta)*(self.Q[a][b][c][d][e]) + self.eta*(reward + self.gamma*(self.Q[h][i][j][k])[new_action])
		self.last_action = new_action
		self.last_state  = new_state
		#if new_state['score'] > self.last_state['score']:
			#print(new_state['score'])

		#print(new_state)
		return self.last_action

	def reward_callback(self, reward):
		'''This gets called so you can see what reward you get.'''

		self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''
	for ii in trange(iters):
		# Make a new monkey object.
		swing = SwingyMonkey(sound=False,                  # Don't play sounds.
							 text="Epoch %d" % (ii),       # Display the epoch on screen.
							 tick_length = t_len,          # Make game ticks super fast.
							 action_callback=learner.action_callback,
							 reward_callback=learner.reward_callback)

		# Loop until you hit something.
		while swing.game_loop():
			pass
		
		# Save score history.
		hist.append(swing.score)

		# Reset the state of the learner.
		learner.reset()
	pg.quit()
	return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 100, 0)

	# Save history. 
	np.save('hist',np.array(hist))
	print(max(hist))
	print(np.mean(hist))
	plt.plot(range(len(hist)), hist)
	plt.show()


