import gym
import torch
from torch import nn
import torchvision
from collections import deque, namedtuple
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple("Transition", ["state", "action", "reward", "nextState"])

MEMORY_SIZE = 10000
replayMemory = deque(maxlen=MEMORY_SIZE)

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.convNet = nn.Sequential(
			nn.Conv2d(3, 16, 3),
			nn.LeakyReLU(),
			nn.Conv2d(16, 32, 3),
			nn.LeakyReLU(),
			nn.Flatten()
		)
		numFeatures = self.getConvOutputSize((3, 210, 160))
		self.linearNet = self.createLinearNet(numFeatures)
	
	def createLinearNet(self, inputFeatures):
		linearNet = nn.Sequential(
			nn.Linear(inputFeatures, 32),
			nn.LeakyReLU(),
			nn.Linear(32, 16),
			nn.LeakyReLU(),
			nn.Linear(16, 4)
		)
		return linearNet

	def getConvOutputSize(self, inputShape):
		dummyInput = torch.zeros(1, *inputShape)
		output = self.convNet(dummyInput)
		return int(output.shape[1])

	def forward(self, x):
		linearFeatures = self.convNet(x)
		return self.linearNet(linearFeatures)

QNetwork = DQN()
QNetwork.to(device)

def fmtObservation(obs):
	obs = torch.from_numpy(obs/255).type(torch.float32)
	obs = torch.permute(obs, (2, 0, 1))
	return obs

def chooseAction(state, QNet, randomChance):
	randomNum = random.random()
	bestAction = -1
	if randomNum < randomChance:
		bestAction = random.randrange(0, 4)
	else:
		predictedRewards = QNet(torch.unsqueeze(state, 0))
		predictedRewardsNumpy = predictedRewards.detach().numpy() 
		bestAction = np.argmax(predictedRewardsNumpy)
	return bestAction

env = gym.make('ALE/Breakout-v5', full_action_space=False)

def collectTransitions(randomChance):
	observation = env.reset()
	done = False
	step = 0
	while not done:
		step += 1
		currentState = observation
		currentState = fmtObservation(currentState)

		action = chooseAction(currentState, QNetwork, randomChance)

		observation, reward, done, info = env.step(action)

		newState = observation
		newState = fmtObservation(newState)

		transition = Transition(currentState, action, reward, newState)
		replayMemory.append(transition)

def trainStep(lossFunction, optimizer, batchSize):
	batch = random.sample(replayMemory, batchSize)
	
	states, actions, rewards, nextStates = zip(*batch)
	states = torch.stack(states)
	rewards = torch.from_numpy(np.array(rewards))
	nextStates = torch.stack(nextStates)
	actions = torch.unsqueeze(torch.Tensor(actions), 1).to(torch.int64)
	
	predictedRewards = QNetwork(states)


	with torch.no_grad():
		futurePredictions = QNetwork(nextStates)

	maxFutureRewards = torch.max(futurePredictions, 1)[0]

	maxFutureRewards *= 0.99
	maxFutureRewards += rewards

	predictedRewards = torch.gather(predictedRewards, 1, actions)

	predictedRewards = torch.squeeze(predictedRewards, 1)

	optimizer.zero_grad()
	loss = lossFunction(predictedRewards, maxFutureRewards)
	loss.backward()
	for param in QNetwork.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()
	return loss

def evaluateAgent(numGames, QNet, renderGames=False):
	print("Evaluating agent, Render {}".format(renderGames))
	avgReward = 0.
	for i in range(numGames):
		print("Evaluation game {}".format(i))
		totalReward = 0.
		observation = env.reset()
		done = False
		maxStep = 2000
		step = 0
		while not done and step <= maxStep:
			step += 1
			currentState = observation
			currentState = fmtObservation(currentState)
			action = chooseAction(currentState, QNet, 0.)
			observation, reward, done, info = env.step(action)
			totalReward += reward
		avgReward += totalReward
	avgReward /= numGames
	return avgReward


BATCH_SIZE = 32
STEPS_PER_TRAINSTEP = 5
EVALUATE_EVERY = 25
RENDER_EVERY = 100

#lossFunc = torch.nn.MSELoss()
lossFunc = torch.nn.SmoothL1Loss()
optimizer = torch.optim.RMSprop(QNetwork.parameters())

bestSoFar = DQN()
bestReward = 0

randomChance = 1.

if __name__ == "__main__":
	i = 0
	while True:
		print("Collecting transitions, randomChance is {}".format(randomChance))
		collectTransitions(randomChance)
		randomChance *= 0.999
		for _ in range(STEPS_PER_TRAINSTEP):
			print("Running train step")
			trainStep(lossFunc, optimizer, BATCH_SIZE)
		if i%EVALUATE_EVERY == 0:
			avgReward = evaluateAgent(1, QNetwork, False)
			if avgReward > bestReward:
				bestReward = avgReward
				bestSoFar.load_state_dict(QNetwork.state_dict())
			print("Train Step {}: Average reward is {}".format(i, avgReward))
			print("Best Reward so far: {}".format(bestReward))
		if i%RENDER_EVERY == 0:
			evaluateAgent(1, bestSoFar, True)
		i += 1
