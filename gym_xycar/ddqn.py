import gym
import random
import datetime
import os
import numpy as np
from collections import deque

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

env = gym.make('gym_xycar:xycar-v0')

algorithm="Double_DQN"

state_size=14
action_size=env.action_space.n

load_model=True
train_mode=True

batch_size=64
mem_maxlen=10000

discount_factor=0.99
learning_rate=0.00025

skip_frame=1
stack_frame=1

start_train_step=1000
run_step=100000
test_step=10000

target_update_step=1000
print_episode=10
check_reward=0

epsilon_init=0.9
epsilon_min=0.1

date_time=datetime.datetime.now().strftime("%Y%m%d")

save_path="./saved_models/"+date_time+"/"
load_path="./map_3action_256x256_ep9.pth"

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, network_name):
        super(DQN, self).__init__()
        input_size=state_size*stack_frame

        self.fc1=nn.Linear(input_size, 256)
        self.fc2=nn.Linear(256, 256)
        self.fc3=nn.Linear(256, action_size)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

class DQNAgent():
    def __init__(self, model, target_model, optimizer):
        self.model=model
        self.target_model=target_model
        self.optimizer=optimizer

        self.memory=deque(maxlen=mem_maxlen)
        self.obs_set=deque(maxlen=skip_frame*stack_frame)

        self.epsilon=epsilon_init

        self.update_target()

        if load_model==True:
            self.model.load_state_dict(torch.load(load_path, map_location=device))
            print("Model is loaded from {}".format(load_path))


    def skip_stack_frame(self, obs):
        self.obs_set.append(obs)

        state=np.zeros([state_size*stack_frame])

        for i in range(stack_frame):
            state[state_size*i:state_size*(i+1)]=self.obs_set[-1-(skip_frame*i)]

        return state


    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def get_action(self, state):
        if train_mode:
            if self.epsilon>np.random.rand():
                return np.random.randint(0, action_size)

            else:
                with torch.no_grad():
                    Q=self.model(torch.FloatTensor(state).unsqueeze(0).to(device))

                    return np.argmax(Q.cpu().detach().numpy())

        else:
            with torch.no_grad():
                Q=self.model(torch.FloatTensor(state).unsqueeze(0).to(device))

                return np.argmax(Q.cpu().detach().numpy())


    def save_model(self, load_model, train_mode):
        now=datetime.datetime.now().strftime("%H-%M-%S")

        if not load_model and train_mode:
            if not os.path.exists(os.path.join(save_path, algorithm)):
                print('make directory')
                os.makedirs(save_path+algorithm)

            torch.save(self.model.state_dict(), save_path+algorithm+'/'+now+'model.pth')

            print("Save Model: {}".format(save_path+algorithm+'/'+now+'model.pth'))

        elif load_model and train_mode:
            if not os.path.exists(os.path.join(save_path, algorithm)):
                print('make directory')
                os.makedirs(save_path+algorithm)

            torch.save(self.model.state_dict(), save_path+algorithm+'/'+now+'model.pth')

            print("Save Model: {}".format(save_path+algorithm+'/'+now+'model.pth'))


    def train_model(self):
        batch=random.sample(self.memory, batch_size)

        state_batch=torch.FloatTensor(np.stack([b[0] for b in batch], axis=0)).to(device)
        action_batch=torch.FloatTensor(np.stack([b[1] for b in batch], axis=0)).to(device)
        reward_batch=torch.FloatTensor(np.stack([b[2] for b in batch], axis=0)).to(device)
        next_state_batch=torch.FloatTensor(np.stack([b[3] for b in batch], axis=0)).to(device)
        done_batch=torch.FloatTensor(np.stack([b[4] for b in batch], axis=0)).to(device)
        
        eye=torch.eye(action_size).to(device)
        one_hot_action=eye[action_batch.view(-1).long()]
        #print(one_hot_action)
        q=(self.model(state_batch)*one_hot_action).sum(1)
        #print(self.model(state_batch), q)

        with torch.no_grad():
            max_Q=torch.max(q).item()
            next_q=self.target_model(next_state_batch)
            #print(next_q, next_q.max(1).values)
            #print(reward_batch)
            target_q=reward_batch+next_q.max(1).values*(discount_factor*(1-done_batch))
            
        #print(q[:10], target_q[:10])
        loss=F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), max_Q


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict()) 


if __name__=='__main__':
    model=DQN("main").to(device)
    target_model=DQN("target").to(device)
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)

    agent=DQNAgent(model, target_model, optimizer)

    model.train()

    step=0
    episode=0
    reward_list=[]
    loss_list=[]
    max_Q_list=[]

    while step<run_step+test_step+start_train_step:
        obs=env.reset()

        episode_rewards=0
        done=False

        for i in range(skip_frame*stack_frame):
            agent.obs_set.append(obs)

        state=agent.skip_stack_frame(obs)

        while not done:
            env.render()

            if step==run_step:
                train_mode=False
                model.eval()

                print('finish train')
            
            action=agent.get_action(state)
            next_obs, reward, done, _=env.step(action)
            episode_rewards+=reward

            next_state=agent.skip_stack_frame(next_obs)

            if train_mode:                    
                agent.append_sample(state, action, reward, next_state, done)

                if step>start_train_step:
                    if agent.epsilon>epsilon_min:
                        agent.epsilon-=1.0/run_step
                    #print(agent.epsilon, 1.0/run_step)

                    loss, maxQ = agent.train_model()
                    loss_list.append(loss)
                    max_Q_list.append(maxQ)
                else:
                    if step%1000==0:
                        print('sampling...')
                
                if step==start_train_step:
                    print("start train")

                if step%target_update_step==0:
                    agent.update_target()

            else:
                agent.epsilon=0.0

            state=next_state
            step+=1

        print("episode reward: {}".format(episode_rewards))
        reward_list.append(episode_rewards)
        episode+=1

        if episode%print_episode==0 and episode!=0:
            #print(reward_list, loss_list, max_Q_list)
            #print(np.mean(reward_list), np.mean(loss_list), np.mean(max_Q_list))
            
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f} / maxQ: {:.2f} / epsilon: {:.4f}".format
                (step, episode, np.mean(reward_list), np.mean(loss_list), np.mean(max_Q_list), agent.epsilon))

            reward_list=[]
            loss_list=[]
            max_Q_list=[]

        if step>start_train_step and train_mode and episode_rewards-check_reward>50:
            agent.save_model(load_model, train_mode)

            check_reward=episode_rewards

    agent.save_model(load_model, train_mode)
    env.close()