#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch

class ReplayBuffer:
    """
    Experience Replay를 위한 버퍼 구현
    Args:
        obs_dim (int): observation의 dimension.
        act_dim (int): action의 dimension.
        size (int): 버퍼의 최대 데이터 저장한 수.
    Attributes:
        ptr (int): 다음 데이터를 저장할 인덱스.
        size (int): 현재 저장된 데이터 수.
        max_size (int): 버퍼의 최대 데이터 저장한 수.
        obs_buf (ndarray, dtype=float32, size=(max_size, obs_dim)): obs 저장 배열
        obs2_buf (ndarray, dtype=float32, size=(max_size, obs_dim)): next_obs 저장 배열
        act_buf (ndarray, dtype=float32, size=(max_size, act_dim)): action 저장 배열
        rew_buf (ndarray, dtype=float32, size=(max_size,)): reward 저장 배열
        done_buf (ndarray, dtype=float32, size=(max_size,)): done 저장 배열
    """

    def __init__(self, obs_dim, act_dim, size):
        self.ptr, self.size, self.max_size = 0, 0, size
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        

    def store(self, obs, act, rew, next_obs, done):
        """
        obs, act, rew, next_obs, done을 버퍼에 저장.
        Input:
            obs (ndarray, dtype=float32, size=(obs_dim,): observation
            act (ndarray, dtype=float32, size=(act_dim,): action
            rew (float): reward
            obs2 (ndarray, dtype=float32, size=(obs_dim,): next_obs
            done (bool): done
        """
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        """
        버퍼에서 batch_size 수 만큼 무작위로 추출.
        Input:
            batch_size (int): 추출한 샘플 수.
        Returns:
            List of len=5 containing:
                FloatTensor, dtype=float32, shape=(batch_size, obs_dim): obs batch
                FloatTensor, dtype=float32, shape=(batch_size, act_dim): action batch
                FloatTensor, dtype=float32, shape=(batch_size,): reward batch
                FloatTensor, dtype=float32, shape=(batch_size, obs_dim): next_obs batch
                FloatTensor, dtype=float32, shape=(batch_size,): done batch
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return list(map(torch.from_numpy, [self.obs_buf[idxs], self.act_buf[idxs], self.rew_buf[idxs], self.obs2_buf[idxs], self.done_buf[idxs]]))