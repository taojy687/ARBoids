import torch
import torch.nn.functional as F
import numpy as np
import copy
from RL.Networks import *
        
class SAC():
    def __init__(self, 
                 cfg,
                 feature1_dim,
                 feature2_dim,
                 action_dim, 
                 max_action=1.0, 
                 adaptive=False,
                 attacker=False,
                 device=torch.device('cpu'),
                 ):
        
        self.batch_size = cfg.rl.batch_size
        self.GAMMA = cfg.rl.GAMMA
        self.TAU = cfg.rl.TAU
        self.lr = cfg.rl.learning_rate
        hidden_dim = cfg.rl.hidden_dim
        
        self.device = device
        self.max_action = max_action
        self.adaptive = adaptive

        self.adaptive_alpha = True  # whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = -dim(A)
            self.target_entropy = -action_dim
            # learn log_alpha instead of alpha to ensure that alpha > 0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().to(self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2
        
        if self.adaptive:
            self.actor = ActorAdap(feature1_dim, feature2_dim, action_dim, hidden_dim, max_action).to(device)
        else:
            if attacker:
                self.actor = ActorAtt(feature1_dim, feature2_dim, action_dim, hidden_dim, max_action).to(device)
            else:
                self.actor = ActorSAC(feature1_dim, feature2_dim, action_dim, hidden_dim, max_action).to(device)

        if attacker:
            self.critic = CriticAtt(feature1_dim, feature2_dim, action_dim, hidden_dim).to(device)
        else:
            self.critic = CriticSAC(feature1_dim, feature2_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)

    def choose_action(self, s, deterministic=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).to(self.device)
            a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
            return a.cpu().numpy().flatten()
    
    def learn(self, replay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = replay_buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)
            # compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)
        
        # compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False
        
        # Compute actor loss
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True
        
        # Update alpha
        if self.adaptive_alpha:
            log_pi = log_pi.to(torch.device('cpu'))
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().to(self.device)
        
        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1.0 - self.TAU) * target_param.data)

    def save(self, dir_actor):
        torch.save(self.actor.state_dict(), dir_actor)
        print('[INFO] Saving Actor Model')

    def load(self, dir_actor):
        self.actor.load_state_dict(torch.load(dir_actor, map_location='cpu', weights_only=True))
        print('[INFO] Loading Actor Model')

    def save_all(self, dir_actor, dir_critic, dir_critic_target):
        torch.save(self.actor.state_dict(), dir_actor)
        torch.save(self.critic.state_dict(), dir_critic)
        torch.save(self.critic_target.state_dict(), dir_critic_target)
        print('[INFO] Saving Actor and Critic Models')
    
    def load_all(self, dir_actor, dir_critic, dir_critic_target):
        self.actor.load_state_dict(torch.load(dir_actor, map_location='cpu', weights_only=True))
        self.critic.load_state_dict(torch.load(dir_critic, map_location='cpu', weights_only=True))
        self.critic_target.load_state_dict(torch.load(dir_critic_target, map_location='cpu', weights_only=True))
        print('[INFO] Loading Actor and Critic Models')

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros(self.max_size)
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros(self.max_size)

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw 
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, device=torch.device('cpu')):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float).to(device)
        batch_a = torch.tensor(self.a[index], dtype=torch.float).to(device)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).unsqueeze(-1).to(device)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(device)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float).unsqueeze(-1).to(device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw

