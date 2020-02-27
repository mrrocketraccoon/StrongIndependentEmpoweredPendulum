import pickle
from collections import namedtuple

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import matplotlib.pyplot as plt
import numpy as np

env = gym.make("Pendulum-v0")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

NUM_ACTIONS = 5
NUM_THETAS = 4
NUM_THETA_DOTS = 3
actions = [i * 4. / (NUM_ACTIONS-1) - 2. for i in range(NUM_ACTIONS)]
thetas = torch.FloatTensor(np.linspace(-np.pi, np.pi - (np.pi * 2) / NUM_THETAS, NUM_THETAS))
theta_dots = torch.FloatTensor(np.linspace(-8., 8., NUM_THETA_DOTS))



def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class PredefinedDynamics(nn.Module):
    def __init__(self):
        super(PredefinedDynamics, self).__init__()
        self.g = 10.0
        self.l = 1.0
        self.m = 1.0
        self.dt = 0.05

    def forward(self, u, s):
        pos = torch.atan2(s[:, 1:2], s[:, 0:1])
        vel = s[:, 2:3]
        u = u.clamp(min=-2.0, max=2.0)

        torque = -3. * self.g / (2. * self.l) * torch.sin(pos + np.pi) + 3. / (self.m * self.l ** 2.) * u[:, 0:1]

        reward = -(angle_normalize(pos) ** 2 + 0.1 * vel ** 2 + 0.001 * torque ** 2)
        vel = vel + self.dt * torque
        pos = pos + vel * self.dt
        vel = vel.clamp(min=-8.0, max=8.0)

        s_next = torch.cat([torch.cos(pos), torch.sin(pos), vel], dim=-1)
        return s_next, reward


dynamics = PredefinedDynamics()

NUM_ACTIONS = 5
NUM_THETAS = 4
NUM_THETA_DOTS = 3
actions = [i * 4. / (NUM_ACTIONS-1) - 2. for i in range(NUM_ACTIONS)]
thetas = torch.FloatTensor(np.linspace(-np.pi, np.pi - (np.pi * 2) / NUM_THETAS, NUM_THETAS))
theta_dots = torch.FloatTensor(np.linspace(-8., 8., NUM_THETA_DOTS))

rewards = np.ones((NUM_THETA_DOTS, NUM_THETAS)) * -1000
actions_taken = np.ones((NUM_THETA_DOTS, NUM_THETAS)) * -1000

for i, t in enumerate(thetas):
    for j, td in enumerate(theta_dots):
        s = torch.FloatTensor([np.cos(t), np.sin(t), td]).view(-1,  3)
        for k, a in enumerate(actions):
            a = torch.FloatTensor([a]).view(-1, 1)
            s_forward, r = dynamics(a, s)
            if r > rewards[j, i]:
                rewards[j, i] = r
                actions_taken[j, i] = a

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
im = axes.imshow(rewards, interpolation="None")
axes.set_title('Rewards for states')
axes.set_xlabel(r'$\theta$ [rad]')
axes.set_ylabel(r'$\dot{\theta}$ [rad/s]')
fig.colorbar(im)
plt.show()