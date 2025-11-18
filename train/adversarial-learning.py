import torch
import os
import numpy as np
from RL.SAC import SAC, ReplayBuffer
from Simu.TADgame import TADEnv
from Utils.config import load_config
from Utils.manager import ExperimentManager

def evaluate(def_agent, 
             att_agent,
             LearningSide: str = 'Def',
             Round: int = 1,
             defender_num : int = 3, 
             agility : float = 2.0,
             ):
    '''
        Evaluate current policies of the defenders and attacker
        
        Return:
            Defender SR, Defender collision, Attacker SR, Mean reward
    '''
    with torch.no_grad():
        env = TADEnv(defender_num,
                    LearningSide=LearningSide)
        
        def_win_num = 0
        att_win_num = 0
        def_col_num = 0
        reward = 0.0
        n = 50

        for _ in range(n):
            def_s, att_s = env.reset(agility, noisy_agility=False)
            done = False
            while not done:
                def_a = def_agent.choose_action(def_s, True)
                def_a = def_a.reshape(env.defender_num, -1)

                if LearningSide == 'Att' or Round > 1:
                    att_a = att_agent.choose_action(att_s, True)
                else:
                    att_a = None

                def_s_, _, done, att_s_ = env.step(def_a, 'AdaRes', att_a)
                def_s, att_s = def_s_, att_s_

            if done == 1:
                att_win_num += 1
            elif done == 2:
                def_col_num += 1
            elif done > 2:
                def_win_num += 1
            reward += env.Rewards.mean() / n

    print('Defender SR =', def_win_num / n)
    print('Defender Col =', def_col_num / n)
    print('Attacker SR =', att_win_num / n)
    print('Mean Reward =', reward)
    return def_win_num / n, def_col_num / n, att_win_num / n, reward

def main(cfg, exp: ExperimentManager, device=torch.device('cuda:0')):

    defender_num = cfg.agent.defender_num
    Round = cfg.agent.round_num
    LearningSide = cfg.agent.learning_side

    warm_steps = cfg.training.warm_steps
    total_steps = cfg.training.total_steps
    eval_interval = cfg.training.eval_interval

    agility = 2.0
    
    env = TADEnv(defender_num,
                 LearningSide=LearningSide)

    # Attacker Agent
    state_dim = 8
    action_dim = 2
    feature1_dim = 2
    feature2_dim = 6
    att_agent = SAC(cfg, feature1_dim, feature2_dim, action_dim, 
                    adaptive=False, attacker=True, device=device)
    # Loading Models
    if Round > 1:
        load_path = exp.exp_dir + '/att' + str(Round - 1)
        load_att_actor = load_path + '/att-actor' + str(Round - 1) + '.pth'
        load_att_critic = load_path + '/att-critic' + str(Round - 1) + '.pth'
        load_att_critic_target = load_path + '/att-critic-target' + str(Round - 1) + '.pth'
        att_agent.load_all(load_att_actor, load_att_critic, load_att_critic_target) # CHANGING
        # att_agent.load(load_att_actor)

    if LearningSide == 'Att':
        replay_buffer = ReplayBuffer(state_dim, action_dim)
        save_path = exp.exp_dir + '/att' + str(Round)
        os.makedirs(save_path, exist_ok=True)
        dir_actor = save_path + '/att-actor' + str(Round) + '.pth'
        dir_critic = save_path + '/att-critic' + str(Round) + '.pth'
        dir_critic_target = save_path + '/att-critic-target' + str(Round) + '.pth'

    # Defender Agent
    state_dim = env.state_dim
    action_dim = 3
    feature1_dim = env.feature1_dim
    feature2_dim = env.feature2_dim
    def_agent = SAC(cfg, feature1_dim, feature2_dim, action_dim, 
                    adaptive=True, attacker=False, device=device)
    
    # Loading Models
    if Round > 1 or LearningSide == 'Att':
        index = Round if LearningSide == 'Att' else Round - 1
        load_path = exp.exp_dir + '/def' + str(index)
        load_def_actor = load_path + '/def-actor' + str(index) + '.pth'
        load_def_critic = load_path + '/def-critic' + str(index) + '.pth'
        load_def_critic_target = load_path + '/def-critic-target' + str(index) + '.pth'
        def_agent.load_all(load_def_actor, load_def_critic, load_def_critic_target) # CHANGING
        # def_agent.load(load_def_actor)

    if LearningSide == 'Def':
        replay_buffer = ReplayBuffer(state_dim, action_dim)
        save_path = exp.exp_dir + '/def' + str(Round)
        os.makedirs(save_path, exist_ok=True)
        dir_actor = save_path + '/def-actor' + str(Round) + '.pth'
        dir_critic = save_path + '/def-critic' + str(Round) + '.pth'
        dir_critic_target = save_path + '/def-critic-target' + str(Round) + '.pth'
        print('[INFO] Defender is Learning')
    else:
        print('[INFO] Attacker is Learning')
    print('[INFO] Checkpoint Path:', save_path)

    train_steps, eval_num = 0, 0
    while train_steps < total_steps:
        def_s, att_s = env.reset(agility, noisy_agility=True)
        done = False

        while not done:
            if train_steps < warm_steps and Round == 1:
                if LearningSide == 'Def':
                    def_action = np.random.uniform(-1.0, 1.0, (env.defender_num, action_dim))
                    def_action[:, -1] = def_action[:, -1] * 0.5 + 0.5
                else:
                    def_action = def_agent.choose_action(def_s, False)
                    def_action = def_action.reshape(env.defender_num, action_dim)
                
                att_action = None
            else:
                # Defenders choose actions
                def_action = def_agent.choose_action(def_s, False)
                def_action = def_action.reshape(env.defender_num, action_dim)
                def_action[:, -1] = np.clip(def_action[:, -1] + np.random.uniform(-0.1, 0.1), 0.0, 1.0)

                # Attacker chooses action
                if LearningSide == 'Att' or Round > 1:
                    att_action = att_agent.choose_action(att_s, False)
                else:
                    att_action = None

            def_s_, r, done, att_s_ = env.step(def_action, 'AdaRes', att_action)

            if LearningSide == 'Def':
                for i in range(env.defender_num): 
                    replay_buffer.store(def_s[i], def_action[i], r[i], def_s_[i], bool(done))
            else:
                replay_buffer.store(att_s, env.att_action, r, att_s_, bool(done))

            def_s, att_s = def_s_, att_s_
            train_steps += 1
            
            if train_steps >= warm_steps:
                if LearningSide == 'Def':
                    def_agent.learn(replay_buffer)
                else:
                    att_agent.learn(replay_buffer)

                if train_steps % eval_interval == 0:
                    eval_num += 1
                    def_sr, def_col, att_sr, reward = evaluate(
                        def_agent, att_agent, LearningSide, Round, defender_num, agility)
                    
                    exp.record_metrics(num=eval_num, 
                                       LearningSide=LearningSide,
                                       Round=Round,
                                       Def_SR=def_sr,
                                       Def_Col=def_col,
                                       Att_SR=att_sr,
                                       Reward=reward
                                       )

                    if LearningSide == 'Def':
                        def_agent.save_all(dir_actor, dir_critic, dir_critic_target)
                    else:
                        att_agent.save_all(dir_actor, dir_critic, dir_critic_target)

if __name__ == "__main__":

    cfg = load_config('configs/adversarial.yaml')
    exp = ExperimentManager(
                    cfg, 
                    base_dir='experiments',
                    run_id='al-exp',
                    repeat_idx=None
                    )
    device = torch.device('cuda:0')

    main(cfg, exp, device)

