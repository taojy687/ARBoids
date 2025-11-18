import torch
import numpy as np
from RL.SAC import SAC, ReplayBuffer
from Simu.TADgame import TADEnv
from Utils.config import load_config
from Utils.manager import ExperimentManager

def evaluate(agent, 
             defender_num=3, 
             agility=2.0,
             boid_state=True,
             controller='Res'):
    with torch.no_grad():
        env = TADEnv(defender_num,
                    boid_state)
        
        def_win_num = 0
        reward = 0.0
        n = 50
        for _ in range(n):
            s, _ = env.reset(agility, noisy_agility=False)
            done = False
            while not done:
                a = agent.choose_action(s, True)
                a = a.reshape(env.defender_num, -1)
                s_, r, done, _ = env.step(a, controller)
                s = s_
            if done > 2:
                def_win_num += 1
            reward += env.Rewards.mean() / n
    return def_win_num / n, reward

def main(cfg, exp: ExperimentManager, device=torch.device('cuda:0')):

    defender_num = cfg.agent.defender_num
    adaptive = cfg.agent.adaptive
    residual = cfg.agent.residual
    boid_state = cfg.agent.boid_state
    form_reward = cfg.agent.form_reward
    curriculum = cfg.agent.curriculum

    model_name = cfg.training.model_name
    warm_steps = cfg.training.warm_steps
    total_steps = cfg.training.total_steps
    eval_interval = cfg.training.eval_interval

    # Curriculum learning
    init_agility = cfg.curriculum.init_agility
    eva_agility = cfg.curriculum.eva_agility
    ind_agility = cfg.curriculum.ind_agility

    env = TADEnv(defender_num,
                 boid_state,
                 form_reward)

    state_dim = env.state_dim
    action_dim = env.action_dim
    if adaptive:
        action_dim += 1

    agent = SAC(cfg, env.feature1_dim, env.feature2_dim, action_dim, adaptive=adaptive, device=device)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Controller type
    if residual:
        if adaptive:
            controller = 'AdaRes'
        else:
            controller = 'Res'
    else:
        controller = 'RL'
    
    print('[INFO] Controller type is', controller)

    train_steps, eval_num = 0, 0
    while train_steps < total_steps:
        if curriculum:
            agility = int(4 * train_steps / total_steps) * ind_agility + init_agility
            s, _ = env.reset(agility, noisy_agility=True)
        else:
            agility = eva_agility
            s, _ = env.reset(agility, noisy_agility=False)
        done = False
        while not done:

            if train_steps < warm_steps:
                action = np.random.uniform(-1.0, 1.0, (env.defender_num, action_dim))
                if adaptive:
                    action[:, -1] = action[:, -1] * 0.5 + 0.5
            else:
                action = agent.choose_action(s, False)
                action = action.reshape(env.defender_num, action_dim)
                if adaptive:
                    action[:, -1] = np.clip(action[:, -1] + np.random.uniform(-0.1, 0.1), 0.0, 1.0)
            
            s_, r, done, _ = env.step(action, controller)

            for i in range(env.defender_num): 
                replay_buffer.store(s[i], action[i], r[i], s_[i], bool(done))
            s = s_

            train_steps += 1
            if train_steps >= warm_steps:
                agent.learn(replay_buffer)

                if train_steps % eval_interval == 0:
                    eval_num += 1
                    def_sr, reward = evaluate(agent, defender_num, eva_agility, boid_state, controller)
                    exp.record_metrics(num=eval_num, def_sr=def_sr, reward=reward)
                    exp.save_model(agent, model_name)

if __name__ == "__main__":

    cfg = load_config('configs/train.yaml')

    exp = ExperimentManager(
                    cfg, 
                    base_dir='experiments',
                    run_id='arboids',
                    repeat_idx=2
                    )
    device = torch.device('cuda:0')

    main(cfg, exp, device)

