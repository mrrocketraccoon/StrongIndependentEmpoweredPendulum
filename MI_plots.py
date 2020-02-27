import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

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
    return mutual_information.item()


def EmpowermentPlot(pathplanning, pathsource):
    source_network = Source(n_actions=1, n_states=3)
    source_network.load_state_dict(torch.load(pathsource))
    planning_network = Planning(n_actions=1, n_states=3)
    planning_network.load_state_dict(torch.load(pathplanning))
    dynamics = Dynamics()

    theta = np.linspace(-np.pi, np.pi, 100)
    theta_dot = np.linspace(8., -8, 50)
    values = np.zeros((50, 100))
    actions = np.zeros((50, 100))
    for i, t in enumerate(theta):
        for j, td in enumerate(theta_dot):
            s = np.stack([np.cos(t), np.sin(t), td])
            s_tensor = torch.from_numpy(s)
            MIs = []
            sam = []
            with torch.no_grad():
                for a in range(5):
                    source_action, source_mean, source_log_var = source_network(s_tensor)
                    s_next = np.asarray(dynamics.step(source_action, s))[0]
                    s_next_tensor = torch.from_numpy(s_next)
                    planning_action, planning_mean, planning_log_var = planning_network(s_tensor, s_next_tensor)
                    MIs.append(mutual_info(source_action, source_mean, source_log_var, planning_action, planning_mean, planning_log_var))
                    sam.append(source_action.item())
            mi = sum(MIs)/10
            if mi > 2:
                values[j, i] = 2
                actions[j, i] = 2

            elif mi < 0:
                values[j, i] = 0
                actions[j, i] = 0
            else:
                values[j, i] = sum(MIs) / 10
                actions[j, i] = sum(sam) / 10

        print("Loading Image: %s%complete", i)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    im = axes.imshow(values, interpolation="None")
    axes.set_title('Values of $\hat{I}(z)$')
    axes.set_xlabel('$\theta$ [rad]')
    axes.set_ylabel('$\dot{\theta}$ [rad/s]')
    plt.setp(axes, xticks=[0, 25, 50, 75, 100], xticklabels=['-$\pi$', '-$\pi/2$', '0', '$\pi/2$', '$\pi$'], yticks=[0, 25, 50], yticklabels=['8', '0', '-8'])
    fig.colorbar(im)
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    im = axes.imshow(actions, interpolation="None")
    axes.set_title('Action values')
    axes.set_xlabel('$\theta$ [rad]')
    axes.set_ylabel('$\dot{\theta}$ [rad/s]')
    plt.setp(axes, xticks=[0, 25, 50, 75, 100], xticklabels=['-$\pi$', '-$\pi/2$', '0', '$\pi/2$', '$\pi$'], yticks=[0, 25, 50], yticklabels=['8', '0', '-8'])
    fig.colorbar(im)
    plt.show()

if __name__ == "__main__":
    EmpowermentPlot('Model/planningcontrol19.pkl', 'Model/sourcecontrol19.pkl')