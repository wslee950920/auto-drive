#! /usr/bin/env python
# -*- coding:utf-8 -*-

from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network: 주어진 observation으로부터 어떤 action을 취할지 결정.
    Args:
        obs_dim (int): observation의 dimension.
        act_dim (int): action의 dimension.
        act_limit (float): action의 범위를 -act_limit ~ act_limit로 설정.
            모든 act_dim에 대해 act_limit은 동일하다고 가정한다.
    
    Input:
        obs (FloatTensor, shape=(N, obs_dim)): obs의 batch.
    Output:
        FloatTensor, shape=(N, act_dim): 결정한 action의 batch.
    """

    def __init__(self, obs_dim, act_dim, act_limit):
        "자유롭게 네트워크를 설계해주세요"
        super(Actor, self).__init__()
        self.act_limit = act_limit
        self.layers = nn.Sequential(
            F.relu(nn.Linear(obs_dim, 256)),
            F.relu(nn.Linear(256, 256)),
            nn.Linear(256, obs_dim),
            nn.Tanh()
        )        

    def forward(self, obs):
        return self.act_limit * self.layers(obs)

class QFunc(nn.Module):
    """
    Critic network: Actor가 결정한 action을 Q value로 평가.
    Args:
        obs_dim (int): observation의 dimension.
        act_dim (int): action의 dimension.
        
    Input:
        obs (FloatTensor, shape=(N, obs_dim)): obs의 batch.
        act (FloatTensor, shape=(N, act_dim)): action의 batch.
    Output:
        FloatTensor, shape=(N,): obs, action의 q value batch.
    """

    def __init__(self, obs_dim, act_dim):
        "자유롭게 네트워크를 설계해주세요"
        super(QFunc, self).__init__()
        self.layers = nn.Sequential(
            F.relu(nn.Linear(obs_dim + act_dim, 256))
            F.relu(nn.Linear(256, 256))
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        q = self.layers(torch.cat([obs, act], dim=-1))

        return torch.squeeze(q, -1)

class ActorCritic(nn.Module):
    """
    Actor-Critic Agent
    Args:
        obs_dim (int): observation의 dimension.
        act_dim (int): action의 dimension.
        act_limit (float): action의 범위를 -act_limit ~ act_limit로 설정.
            모든 act_dim에 대해 act_limit은 동일하다고 가정한다.
        pi_lr (float): actor network의 learning rate.
        q_lr (float): critic network의 learning rate.
        gamma (float): q 함수에서 미래 보상에 대한 감쇠율. 0 <= gamma < 1.
            Q(s, a) = reward + gamma * (1 - done) * Q(s', a').
        polyak (float): soft target update 비율. 0 <= polyak < 1.
            W_target =  polyak * W_target + (1 - polyak) * W_original.
    Attributes:
        pi (Actor): Actor network.
        q (QFunc): Critic network.
        target_pi (Actor): Actor의 target network.
        target_q (QFunc): Critic의 target network.
        pi_optimizer : Actor의 optimizer. 여기서는 Adam을 사용한다.
        q_optimizer : Critic의 optimizer. 여기서는 Adam을 사용한다.
        loss_fn (function or Loss Module): Loss를 계산하기 위한 함수 혹은 클래스. 여기서는 MSELoss를 사용.
        
    """

    def __init__(self, obs_dim, act_dim, act_limit, pi_lr, q_lr, gamma, polyak):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit

        # set actor/critic network
        self.pi = Actor(obs_dim, act_dim, act_limit)
        self.q = QFunc(obs_dim, act_dim)

        # set target networks
        network_size = 0
        self.target_pi= deepcopy(self.pi)
        self.target_q = deepcopy(self.q)
        for param in self.target_pi.parameters():
            network_size += np.prod(param.shape)
            param.requires_grad = False
        for param in self.target_q.parameters():
            network_size += np.prod(param.shape)
            param.requires_grad = False

        # set optimizers
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=pi_lr)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=q_lr)

        # set loss function
        self.loss_fn = nn.MSELoss()

        # set decay rate
        self.gamma = gamma
        self.polyak = polyak

        print('\nYou know DDPG? We are no.1 DDPG Trainer!')
        print('\nNumber of parameters: %d\n'%network_size) # 학습할 파라미터의 개수를 출력

    def get_action(self, obs, noise_scale):
        """
        Actor network로 관측값 obs를 통해 action을 얻고 noise를 적용하여 반환한다.

        Input:
            obs (FloatTensor, shape=(obs_dim,)): single observation.
            noise_scale (float): action에 얼마나 큰 noise를 더해줄 것인지. 훈련시 탐색 범위를 넓히기 위해 사용한다.
                평균 0, 표준편차가 1 인 가우시안 분포에서 난수를 추출하고 이를 noise_scale에 곱하여 action에 더해주는 방식이다.
                action += noise_scale * noise ~ N(0, 1)

        Output:
            ndarray, dtype=float32, shape=(act_dim,): 노이즈가 적용된 action.
        """
        with torch.no_grad():
            action = self.pi(obs).cpu().numpy()
            action += noise_scale * np.random.randn(self.act_dim)

            return np.clip(action, -self.act_limit, self.act_limit)

    def compute_loss_q(self, obs, action, reward, next_obs, done):
        """
        Q function의 update를 위한 loss를 계산한다.
        Input:
            obs (FloatTensor, shape=(N, obs_dim): obs batch.
            action (FloatTensor, shape=(N, act_dim): action batch.
            reward (FloatTensor, shape=(N,): reward batch.
            next_obs (FloatTensor, shape=(N, obs_dim): next_obs batch.
            done (FloatTensor, shape=(N,): done batch.
        Output:
            FloatTensor, shape=(): loss 값.
        """
        q = self.q(obs, action)
        next_action = self.target_pi(next_obs)
        next_q = self.target_q(next_obs, next_action)
        target_q = reward + self.gamma * (1 - done) * next_q
        loss_q = self.loss_fn(q, target_q)

        return loss_q

    def compute_loss_pi(self, obs):
        """
        Actor의 update를 위한 loss를 계산한다.
        q_pi를 증가시키는 방향으로 actor network의 parameter를 update하는 것이
        목표이기 때문에 -q_pi로 경사하강법을 한다.
        Input:
            obs (FloatTensor, shape=(N, obs_dim): obs batch.
        Output:
            FloatTensor, shape=(): loss 값.
        Note:
            Q를 계산하는 network은 이 시점에서 parameter가 freeze 되어있어야 한다.(Actor만 훈련이 되도록)
        """
        action = self.pi(obs)
        q_pi = self.q(obs, action)
        return -q_pi.mean()
    
    def update(self, batch):
        """
        Actor와 Critic을 한 step update 한 후, soft target update를 한다.
        Input:
            batch (Iter, len=5): obs, action, reward, next_obs, done 순으로 packing된 컨테이너.
                obs (FloatTensor, shape=(N, obs_dim): obs batch.
                action (FloatTensor, shape=(N, act_dim): action batch.
                reward (FloatTensor, shape=(N,): reward batch.
                next_obs (FloatTensor, shape=(N, obs_dim): next_obs batch.
                done (FloatTensor, shape=(N,): done batch.
        Output:
            float: q의 loss 값.
            float: pi의 loss 값.
        """
        obs, action, reward, next_obs, done = batch

        # update q
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(obs, action, reward, next_obs, done)
        loss_q.backward()
        self.q_optimizer.step()

        # freeze q
        for param in self.q.parameters():
            param.requires_grad = False

        # update pi
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(obs)
        loss_pi.backward()
        self.pi_optimizer.step()

        # unfreeze q
        for param in self.q.parameters():
            param.requires_grad = True

        # update target network
        with torch.no_grad():
            for param, param_target in zip(self.q.parameters(), self.target_q.parameters()):
                param_target.data.mul_(self.polyak)
                param_target.data.add_((1.0 - self.polyak) * param.data)
            for param, param_target in zip(self.pi.parameters(), self.target_pi.parameters()):
                param_target.data.mul_(self.polyak)
                param_target.data.add_((1.0 - self.polyak) * param.data)

        return loss_q.item(), loss_pi.item()