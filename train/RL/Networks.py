import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.nn.utils import weight_norm

# =================================================================== #
#     Network Initialize Functions
# =================================================================== #
def _orthogonal_init(layer, gain=0.1):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer

# =================================================================== #
#     SAC Actor and Critic Networks
# =================================================================== #
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
        self.apply(_orthogonal_init)

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

class CriticSAC(nn.Module):
    def __init__(self, 
                 feature1_dim,
                 feature2_dim, 
                 action_dim, 
                 hidden_dim,
                 ):
        super(CriticSAC, self).__init__()
        self.feature1_dim = feature1_dim
        self.feature2_dim = feature2_dim

        # Activation Function
        self.activation = nn.LeakyReLU()

        # Q1
        self.f1 = nn.Linear(feature1_dim, hidden_dim // 2)
        self.f2 = nn.Linear(feature2_dim, hidden_dim // 2)
        self.t1 = nn.Linear(2, hidden_dim // 4)
        self.a1 = nn.Linear(action_dim, hidden_dim // 4)

        self.l1 = nn.Linear(int(1.5 * hidden_dim), hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.f3 = nn.Linear(feature1_dim, hidden_dim // 2)
        self.f4 = nn.Linear(feature2_dim, hidden_dim // 2)
        self.t2 = nn.Linear(2, hidden_dim // 4)
        self.a2 = nn.Linear(action_dim, hidden_dim // 4)

        self.l4 = nn.Linear(int(1.5 * hidden_dim), hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

        self.apply(_orthogonal_init)

    def forward(self, s, a):
        batch_size = s.shape[0]
        # Q1
        s1 = self.activation(self.f1(s[..., : self.feature1_dim]))
        s2 = self.activation(self.f2(s[..., self.feature1_dim : self.feature1_dim + self.feature2_dim]))
        s3 = s[..., self.feature1_dim+self.feature2_dim:].view(batch_size, -1, 2)
        s3 = self.activation(self.t1(s3)).mean(dim=-2, keepdim=False)
        a1 = self.activation(self.a1(a))
        s_a_1 = torch.concat([s1, s2, s3, a1], dim=-1)

        q1 = self.activation(self.l1(s_a_1))
        q1 = self.activation(self.l2(q1))
        q1 = self.l3(q1)

        # Q2
        s4 = self.activation(self.f3(s[..., : self.feature1_dim]))
        s5 = self.activation(self.f4(s[..., self.feature1_dim : self.feature1_dim + self.feature2_dim]))
        s6 = s[..., self.feature1_dim+self.feature2_dim:].view(batch_size, -1, 2)
        s6 = self.activation(self.t2(s6)).mean(dim=-2, keepdim=False)
        a2 = self.activation(self.a2(a))
        s_a_2 = torch.concat([s4, s5, s6, a2], dim=-1)

        q2 = self.activation(self.l4(s_a_2))
        q2 = self.activation(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
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

        self.apply(_orthogonal_init)

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

class ActorAtt(nn.Module):
    def __init__(self, 
                 feature1_dim,
                 feature2_dim, 
                 action_dim, 
                 hidden_dim,
                 max_action=1.0,
                 ):
        super(ActorAtt, self).__init__()
        self.max_action = max_action
        self.feature1_dim = feature1_dim
        self.feature2_dim = feature2_dim

        # Activation Function
        self.activation = nn.LeakyReLU()

        self.f1 = nn.Linear(feature1_dim, hidden_dim // 2)
        self.f2 = nn.Linear(feature2_dim, hidden_dim // 2)

        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.apply(_orthogonal_init)

    def forward(self, x, deterministic=False, with_logprob=True):
        x1 = self.activation(self.f1(x[..., :self.feature1_dim]))
        x2 = self.activation(self.f2(x[..., self.feature1_dim:]))

        x = torch.concat([x1, x2], dim=-1)
        x = self.activation(self.l1(x))
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

class CriticAtt(nn.Module):
    def __init__(self, 
                 feature1_dim,
                 feature2_dim, 
                 action_dim, 
                 hidden_dim,
                 ):
        super(CriticAtt, self).__init__()
        self.feature1_dim = feature1_dim
        self.feature2_dim = feature2_dim

        # Activation Function
        self.activation = nn.LeakyReLU()

        # Q1
        self.f1 = nn.Linear(feature1_dim, hidden_dim // 2)
        self.f2 = nn.Linear(feature2_dim, hidden_dim // 2)
        self.a1 = nn.Linear(action_dim, hidden_dim // 4)

        self.l1 = nn.Linear(int(1.25 * hidden_dim), hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2
        self.f3 = nn.Linear(feature1_dim, hidden_dim // 2)
        self.f4 = nn.Linear(feature2_dim, hidden_dim // 2)
        self.a2 = nn.Linear(action_dim, hidden_dim // 4)

        self.l4 = nn.Linear(int(1.25 * hidden_dim), hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

        self.apply(_orthogonal_init)

    def forward(self, s, a):
        # Q1
        s1 = self.activation(self.f1(s[..., : self.feature1_dim]))
        s2 = self.activation(self.f2(s[..., self.feature1_dim :]))
        a1 = self.activation(self.a1(a))
        s_a_1 = torch.concat([s1, s2, a1], dim=-1)

        q1 = self.activation(self.l1(s_a_1))
        q1 = self.activation(self.l2(q1))
        q1 = self.l3(q1)

        # Q2
        s3 = self.activation(self.f3(s[..., : self.feature1_dim]))
        s4 = self.activation(self.f4(s[..., self.feature1_dim :]))
        a2 = self.activation(self.a2(a))
        s_a_2 = torch.concat([s3, s4, a2], dim=-1)

        q2 = self.activation(self.l4(s_a_2))
        q2 = self.activation(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
