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
            nn.Linear(obs_dim, act_dim),
            nn.Tanh()
        )        

    def forward(self, obs):
        return self.act_limit * self.layers(obs)

class Critic(nn.Module):
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
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            F.relu(nn.Linear(obs_dim + act_dim, 256))
            F.relu(nn.Linear(256, 256))
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        q = self.layers(torch.cat([obs, act], dim=-1))

        return torch.squeeze(q, -1)