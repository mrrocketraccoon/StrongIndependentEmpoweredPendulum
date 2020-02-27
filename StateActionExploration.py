import numpy as np
import gym
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

env = gym.make("Pendulum-v0")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
x = 0

def EmpowermentPlot(planning_network, source_network, dynamics):
    theta = np.linspace(-np.pi, np.pi, 100)
    theta_dot = np.linspace(8., -8, 50)
    values = np.zeros((50, 100))
    actions = np.zeros((50, 100))
    planning = np.zeros((50, 100))
    source = np.zeros((50, 100))

    for i, t in enumerate(theta):
        for j, td in enumerate(theta_dot):
            s = np.stack([np.cos(t), np.sin(t), td])
            s_tensor = torch.from_numpy(s)
            MIs = []
            sam = []
            plannings = []
            sources = []
            with torch.no_grad():
                for a in range(5):
                    source_action, source_mean, source_log_var = source_network(s_tensor)
                    s_next = np.asarray(dynamics.step(source_action, s))[0]
                    s_next_tensor = torch.from_numpy(s_next)
                    planning_action, planning_mean, planning_log_var = planning_network(s_tensor, s_next_tensor)
                    plannings.append(ll_gaussian(source_action, planning_mean, planning_log_var))
                    sources.append(ll_gaussian(source_action, source_mean, source_log_var))
                    MIs.append(mutual_info(source_action, source_mean, source_log_var, planning_action, planning_mean, planning_log_var))
                    sam.append(source_action.item())
            mi = sum(MIs)/10
            if mi > 2:
                values[j, i] = 2
                actions[j, i] = 2
                planning[j, i] = 2
                source[j, i] = 2

            elif mi < 0:
                values[j, i] = 0
                actions[j, i] = 0
                planning[j, i] = 2
                source[j, i] = 2

            else:
                values[j, i] = sum(MIs) / 10
                actions[j, i] = sum(sam) / 10
                planning[j, i] = sum(plannings)/10
                source[j, i] = sum(sources)/10


    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    im = axes.imshow(values, interpolation="None")
    axes.set_title('Values of $\hat{I}(z)$')
    axes.set_xlabel('$\theta$ [rad]')
    axes.set_ylabel('$\dot{\theta}$ [rad/s]')
    #plt.setp(axes, xticks=[0, 25, 50, 75, 100], xticklabels=['-$\pi$', '-$\pi/2$', '0', '$\pi/2$', '$\pi$'], yticks=[0, 25, 50], yticklabels=['8', '0', '-8'])
    fig.colorbar(im)
    plt.savefig("Pics/mi%s.png" % x)
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    im = axes.imshow(actions, interpolation="None")
    axes.set_title('Action values')
    axes.set_xlabel('$\theta$ [rad]')
    axes.set_ylabel('$\dot{\theta}$ [rad/s]')
    #plt.setp(axes, xticks=[0, 25, 50, 75, 100], xticklabels=['-$\pi$', '-$\pi/2$', '0', '$\pi/2$', '$\pi$'], yticks=[0, 25, 50], yticklabels=['8', '0', '-8'])
    fig.colorbar(im)
    plt.savefig("Pics/actions%s.png" % x)
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    im = axes.imshow(planning, interpolation="None")
    axes.set_title('Planning Component')
    axes.set_xlabel('$\theta$ [rad]')
    axes.set_ylabel('$\dot{\theta}$ [rad/s]')
    #plt.setp(axes, xticks=[0, 25, 50, 75, 100], xticklabels=['-$\pi$', '-$\pi/2$', '0', '$\pi/2$', '$\pi$'], yticks=[0, 25, 50], yticklabels=['8', '0', '-8'])
    fig.colorbar(im)
    plt.savefig("Pics/planningcomp%s.png" % x)
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    im = axes.imshow(source, interpolation="None")
    axes.set_title('Source Component')
    axes.set_xlabel('$\theta$ [rad]')
    axes.set_ylabel('$\dot{\theta}$ [rad/s]')
    #plt.setp(axes, xticks=[0, 25, 50, 75, 100], xticklabels=['-$\pi$', '-$\pi/2$', '0', '$\pi/2$', '$\pi$'], yticks=[0, 25, 50], yticklabels=['8', '0', '-8'])
    fig.colorbar(im)
    plt.savefig("Pics/sourcecomp%s.png" % x)
    plt.show()


class Dynamics():
    def __init__(self, g=10.0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.m = 1.
        self.l = 1.
        high = np.array([1., 1., self.max_speed])
    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

    def _get_obs(self):
        sin_theta, cos_theta, thetadot = self.state
        return np.array([sin_theta, cos_theta, thetadot],dtype=float)

    def step(self,u, state):
        th = np.arctan2(state[1], state[0])
        thdot = state[2]
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = self.angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*(u-0.05*thdot)) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([np.cos(newth), np.sin(newth), newthdot],dtype=float)
        return self._get_obs(), -costs, False, {}


class Source(nn.Module):
    def __init__(self, n_actions, n_states):
        super().__init__()

        self.source_mu = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions)
        )
        self.source_log_var = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)+ 1e-5
        eps = torch.rand_like(std)
        return mu + std*eps

    def forward(self, s):
        s = s.float().unsqueeze(0)
        mu = self.source_mu(s)
        log_var = self.source_log_var(s)
        act = self.reparameterize(mu, log_var)
        return act, mu, log_var



class Planning(nn.Module):
    def __init__(self, n_actions, n_states):
        super().__init__()
        self.fcs = nn.Linear(n_states, 128)
        self.fcs_ = nn.Linear(n_states, 128)
        self.planning_mu = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions)
        )
        self.planning_log_var = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)+ 1e-5
        eps = torch.rand_like(std)
        return mu + std*eps

    def forward(self, s, s_next):
        s = s.float().unsqueeze(0)
        s_next = s_next.float().unsqueeze(0)
        s = F.relu(self.fcs(s))
        s_next = F.relu(self.fcs_(s_next))
        s_cat = torch.cat([s, s_next], dim=-1)
        mu = self.planning_mu(s_cat)
        log_var = self.planning_log_var(s_cat)
        act = self.reparameterize(mu, log_var)
        return act, mu, log_var

def ll_gaussian(act, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (act-mu)**2

def mutual_info(source_act, source_mu, source_log_sig, planning_act, planning_mu, planning_log_sig):
    mutual_information = ll_gaussian(source_act, planning_mu, planning_log_sig) - ll_gaussian(source_act, source_mu, source_log_sig)
    return mutual_information.abs()

def kl_divergence(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())

class Policy(nn.Module):
    def __init__(self, n_actions, n_states):
        super().__init__()

        self.source_mu = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions)
        )
        self.source_log_var = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)+ 1e-5
        eps = torch.rand_like(std)
        return mu + std*eps

    def forward(self, s):
        s = s.float().unsqueeze(0)
        mu = self.source_mu(s)
        log_var = self.source_log_var(s)
        act = self.reparameterize(mu, log_var)
        torch.clamp(mu, -2.0, 2.0)
        torch.clamp(act, -2.0, 2.0)
        return (act.item(),), mu, log_var


forward_dynamics = Dynamics()
source_network = Source(n_actions=1, n_states=3)
planning_network = Planning(n_actions=1, n_states=3)
policy = Policy(n_actions=1, n_states=3)
source_optimizer = optim.Adam(source_network.parameters(), lr=1e-4)
planning_optimizer = optim.Adam(planning_network.parameters(), lr=1e-4)
policy_optimizer = optim.Adam(policy.parameters(), lr=1e-4)

EPOCHS = 800
epoch = 0
M = 10
T = 200
precision = 0.05
y = 0
BETA = 5

while epoch < EPOCHS:
    empowerment = 0
    #BETA = np.exp((np.log(160)/755)*epoch + 1.307)
    source_optimizer.zero_grad()
    planning_optimizer.zero_grad()
    trajec_mi = []
    trajec_truemi = []
    for m in range(M):
        score = 0
        state = env.reset()
        insta_mi = []
        insta_truemi = []
        state_tensor = Variable(torch.from_numpy(state), requires_grad=True)
        for t in range(T):
            action, policy_mean, policy_log_var = policy(state_tensor)
            state, reward, done, _ = env.step(action)
            state_ = np.asarray(forward_dynamics.step(action, state))[0]
            state_tensor = Variable(torch.from_numpy(state), requires_grad=True)
            state_tensor_ = Variable(torch.from_numpy(state_), requires_grad=True)
            action_tensor = Variable(torch.tensor(action), requires_grad=True)
            source_action, source_mean, source_log_var = source_network(state_tensor)
            source_state_ = np.asarray(forward_dynamics.step((source_action.item(),), state))[0]
            source_state_tensor_ = Variable(torch.from_numpy(source_state_), requires_grad=True)
            planning_action, planning_mean, planning_log_var = planning_network(state_tensor, source_state_tensor_)
            trueMI = ll_gaussian(source_action, planning_mean, planning_log_var) - ll_gaussian(source_action, source_mean, source_log_var)
            MI = BETA*trueMI + kl_divergence(policy_mean, policy_log_var)
            insta_mi.append(MI)
            insta_truemi.append(trueMI)
        trajec_mi.append(sum(insta_mi))
        trajec_truemi.append(sum(insta_truemi))
    empowerment = -sum(trajec_mi)
    empowerment.backward(retain_graph=True)
    source_optimizer.step()
    planning_optimizer.step()
    policy_optimizer.step()
    trueempowerment = -sum(trajec_truemi)/(M*T)

    if epoch%10 == 0:
        state = env.reset()
        for t in range(T):
            with torch.no_grad():
                action, policy_mean, policy_log_var = policy(state_tensor)
            state, reward, done, _ = env.step(action)
            env.render()

    with torch.no_grad():
        dummy_state = np.array([1, 0, 0])
        dummy_action = (1.0,)
        dummy_state_ = np.asarray(forward_dynamics.step(dummy_action, dummy_state)[0])
        dummy_state_tensor = Variable(torch.from_numpy(dummy_state))
        dummy_state_tensor_ = Variable(torch.from_numpy(dummy_state_))
        dummy_source_action, dummy_source_mean, dummy_source_log_var = source_network(dummy_state_tensor)
        dummy_planning_action, dummy_planning_mean, dummy_planning_log_var = planning_network(dummy_state_tensor, dummy_state_tensor_)
        dummy_policy_action, dummy_policy_mean, dummy_policy_log_var = policy(dummy_state_tensor)

    print('Epoch {}\tEmpowerment: {:.2f}'.format(
                epoch, -trueempowerment.item()))
    print('Test --- state: {}, action:{}, next_state: {},'
          '\n source_action: {}, source_mean: {}, source source_log_var: {}'
          ' \n planning_action: {}, planning_mean: {}, planning_log_var: {}'
          '\n policy_action: {}, policy_mean: {}, policy_log_var: {}'
          .format(dummy_state, dummy_action[0], dummy_state_,
                  dummy_source_action.item(), dummy_source_mean.item(), dummy_source_log_var.item(),
                  dummy_planning_action.item(), dummy_planning_mean.item(), dummy_planning_log_var.item(),
                  dummy_policy_action[0], dummy_policy_mean.item(), dummy_policy_log_var.item()))

    epoch += 1
    BETA += (2000 - 5) / 800
    actual_precision = dummy_action[0] - dummy_planning_mean.item()
    #print(BETA)
    if (abs(actual_precision) < precision):
        precision = abs(actual_precision)
        torch.save(planning_network.state_dict(), 'Model/planninbgood.pkl')
        torch.save(source_network.state_dict(), 'Model/sourcebgood.pkl')
        print("Appropriate source and planning networks. Weights have been saved.")
        EmpowermentPlot(planning_network, source_network, forward_dynamics)

    if epoch % 30 == 0:
        torch.save(planning_network.state_dict(), 'Model/planningcontrol%s.pkl' % y)
        torch.save(source_network.state_dict(), 'Model/sourcecontrol%s.pkl' % y)
        print("Weights for control source and planning networks have been saved.")
        EmpowermentPlot(planning_network, source_network, forward_dynamics)
        print("out of plotting")
        x += 1
        y += 1
env.close()