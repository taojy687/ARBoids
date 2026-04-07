import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class ActorSAC(nn.Module):
    def __init__(self, 
                 feature1_dim,
                 feature2_dim, 
                 action_dim, 
                 hidden_dim,
                 max_action=1.0,
                 ):
        super(ActorSAC, self).__init__()
        self.max_action = max_action
        self.feature1_dim = feature1_dim
        self.feature2_dim = feature2_dim

        # Activation Function
        self.activation = nn.LeakyReLU()

        self.f1 = nn.Linear(feature1_dim, hidden_dim // 2)
        self.f2 = nn.Linear(feature2_dim, hidden_dim // 2)
        self.t1 = nn.Linear(2, hidden_dim // 4)

        self.l1 = nn.Linear(int(1.25 * hidden_dim), hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        batch_size = x.shape[0]
        x1 = self.activation(self.f1(x[..., :self.feature1_dim]))
        x2 = self.activation(self.f2(x[..., self.feature1_dim : self.feature1_dim + self.feature2_dim]))

        x3 = x[..., self.feature1_dim+self.feature2_dim:].view(batch_size, -1, 2)
        x3 = self.activation(self.t1(x3)).mean(dim=-2, keepdim=False)

        x = torch.concat([x1, x2, x3], dim=-1)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        if deterministic:
            a = mean
        else:
            a = dist.rsample()
        
        if with_logprob:
            log_pi = dist.log_prob(a).sum(dim=-1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2*a))).sum(dim=-1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)

        return a, log_pi

    def load(self, modelname):
        self.load_state_dict(torch.load(modelname, map_location='cpu', weights_only=True))

class ActorAdap(nn.Module):
    def __init__(self, 
                 feature1_dim,
                 feature2_dim, 
                 action_dim, 
                 hidden_dim,
                 max_action=1.0,
                 ):
        super(ActorAdap, self).__init__()
        self.max_action = max_action
        self.feature1_dim = feature1_dim
        self.feature2_dim = feature2_dim

        # Activation Function
        self.activation = nn.LeakyReLU()

        self.f1 = nn.Linear(feature1_dim, hidden_dim // 2)
        self.f2 = nn.Linear(feature2_dim, hidden_dim // 2)
        self.t1 = nn.Linear(2, hidden_dim // 4)

        self.l1 = nn.Linear(int(1.25 * hidden_dim), hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim - 1)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim - 1)

        self.a1 = nn.Linear(4, hidden_dim // 4)
        self.adap_layer = nn.Linear(int(1.25 * hidden_dim), 1)

    def forward(self, x, deterministic=False, with_logprob=True):
        batch_size = x.shape[0]
        x1 = self.activation(self.f1(x[..., :self.feature1_dim]))
        x2 = self.activation(self.f2(x[..., self.feature1_dim : self.feature1_dim + self.feature2_dim]))
        a_boids = x[..., self.feature1_dim + self.feature2_dim - 2 : self.feature1_dim + self.feature2_dim]

        x3 = x[..., self.feature1_dim+self.feature2_dim:].view(batch_size, -1, 2)
        x3 = self.activation(self.t1(x3)).mean(dim=-2, keepdim=False)

        x = torch.concat([x1, x2, x3], dim=-1)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        if deterministic:
            a = mean
        else:
            a = dist.rsample()
        
        if with_logprob:
            log_pi = dist.log_prob(a).sum(dim=-1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2*a))).sum(dim=-1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)

        aa = torch.concat([a, a_boids], dim=-1)
        aa = self.activation(self.a1(aa))
        aa = torch.concat([aa, x], dim=-1)
        aa = torch.tanh(self.adap_layer(aa)) * 0.5 + 0.5
        return torch.concat([a, aa], dim=-1), log_pi

    def load(self, modelname):
        self.load_state_dict(torch.load(modelname, map_location='cpu', weights_only=True))

