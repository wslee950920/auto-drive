#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ddpg 에서 변형
Hyper parameters:
    obs_dim (int): observation의 dimension.
    act_dim (int): action의 dimension.
    act_limit (float): action의 범위를 -act_limit ~ act_limit로 설정.
        모든 act_dim에 대해 act_limit은 동일하다고 가정한다.
    pi_lr (float): actor network의 learning rate.
    q_lr (float): critic network의 learning rate.
    act_noise (float): action에 얼마나 큰 noise를 더해줄 것인지. 훈련시 탐색 범위를 넓히기 위해 사용한다.
    gamma (float): q 함수에서 미래 보상에 대한 감쇠율. 0 <= gamma < 1.
        Q(s, a) = reward + gamma * (1 - done) * Q(s', a').
    polyak (float): soft target update 비율. 0 <= polyak < 1.
        W_target =  polyak * W_target + (1 - polyak) * W_original.
    epochs (int): 몇 epoch을 훈련시킬 것인지.
        일정한 step수로 이루어져있으며, epoch 종료시 weight 저장 및 훈련 경과를 보여준다.
    steps_per_epoch (int): 한 epoch를 몇 개의 step으로 할건지. 
    num_test_episodes (int): 테스트시 몇 에피소드를 돌릴 것인지.
    max_ep_len (int): 에피소드의 최대 step. 넘을 경우 자동으로 episode가 끝난다.
    start_steps (int): 무작위 action을 취하는 step 수.
    update_after (int): 몇 스텝 후 훈련을 시작할 것인지.
    update_every (int): 몇 스텝마다 훈련을 할 것인지.
        10으로 설정하면 10 step마다 10번씩 update를 한다.
    batch_size (int): 배치 사이즈.
    replay_size (int): Experience Replay 메모리 크기.
    save_freq (int): weight 저장 주기.
"""

# import stuffs
import numpy as np
import torch

from model.ActorCritic import ActorCritic
from model.ExperienceReplay import ReplayBuffer

# For reproducibility
seed = 0
if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# hyper parameters
obs_dim = 14
act_dim = 3
act_limit = 30

pi_lr = 0.00025
q_lr = 0.00025

act_noise = "fill"

gamma = "fill"
polyak = "fill"

epochs = "fill"
steps_per_epoch = "fill"
num_test_episodes = "fill"
max_ep_len = "fill"
start_steps = "fill"

batch_size = "fill"
replay_size = "fill"

update_after = "fill"
update_every = "fill"

save_freq = "fill"

def test_agent():
    """
    Test Environment에서 학습된 agent의 성능 확인.
    Note:
        Test시 noise는 0으로 설정한다.
    """
    for _ in xrange(num_test_episodes):
        obs, done, ep_rew, ep_len = test_env.reset(0), False, 0, 0

        while not (done or (ep_len == max_ep_len)):
            obs = torch.from_numpy(obs).to(device)
            action = agent.get_action(obs, 0)
            obs, reward, done, _ = test_env.step(action)
            ep_rew += reward
            ep_len += 1

        print('Test result: steps=%d  rewards=%f'%(ep_len, ep_rew))

# main
if __name__ == "__main__":

    # Acceleration with gpu if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env, test_env = gym.make('gym_xycar:xycar-v0'), gym.make('gym_xycar:xycar-v0')
    memory = ReplayBuffer(obs_dim, act_dim, replay_size)
    agent = ActorCritic(obs_dim, act_dim, act_limit, pi_lr, q_lr, gamma, polyak).to(device)

    obs, ep_rew, ep_len = env.reset(), 0, 0
    loss_q_log, loss_pi_log, ep_rew_log, ep_len_log = [], [], [], []

    for t in xrange(1, epochs * steps_per_epoch + 1):
        # start_steps 동안은 무작위 행동을 취하여 공격적으로 탐색을 한다.
        if t <= start_steps:
            action = 2 * act_limit * (np.random.rand(act_dim) - 0.5)
        # 그 이후는 agent의 action에 따른다.
        else:
            action = agent.get_action(torch.from_numpy(obs).to(device), act_noise)

        # Env step
        next_obs, reward, done, _ = env.step(action)

        # store experience
        memory.store(obs, action, reward, next_obs, done)

        obs = next_obs
        ep_rew += reward
        ep_len += 1

        # End of episode
        if done or (ep_len == max_ep_len):
            ep_rew_log.append(ep_rew)
            ep_len_log.append(ep_len)
            obs, ep_rew, ep_len = env.reset(), 0, 0

        # Optimize
        if t % update_every == 0 and memory.size >= update_after:
            for _ in range(update_every):
                batch = [tensor.to(device) for tensor in memory.sample_batch(batch_size)]
                loss_q, loss_pi = agent.update(batch)
                loss_q_log.append(loss_q)
                loss_pi_log.append(loss_pi)

        # End of Epoch
        if t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
            print('<Epoch {}>'.format(epoch))
            print('  Train result: loss_q={:.4f}  loss_pi={:.4f}  rewards={:.3f}  steps={}'.format(
                     np.mean(loss_q_log), np.mean(loss_pi_log), np.mean(ep_rew_log), int(np.mean(ep_len_log))))
            loss_q_log, loss_pi_log, ep_rew_log, ep_len_log = [], [], [], []

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                torch.save(agent.pi.state_dict(), 'weight_{:0>4}.pth'.format(epoch))

            # Test model
            test_agent()