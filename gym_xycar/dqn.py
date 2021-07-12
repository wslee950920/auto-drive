import gym
import random
import datetime
import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

env = gym.make('CartPole-v0')

algorithm="DQN"

state_size=18
action_size=env.action_space.n

load_model=False
train_mode=True

batch_size=32

discount_factor=0.99
learning_rate=0.00025

run_step=40000
test_step=10000

print_episode=10
save_step=20000

epsilon_init=1.0
epsilon_min=0.1

date_time=datetime.datetime.now().strftime("%Y%m%d")

save_path="./saved_models/"+date_time+'/'
load_path="."

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1=nn.Linear(state_size, 512)
        self.fc2=nn.Linear(512, 512)
        self.fc3=nn.Linear(512, action_size)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

class DQNAgent():
    def __init__(self, model, optimizer):
        self.model=model
        self.optimizer=optimizer

        self.epsilon=epsilon_init

        if load_model==True:
            self.model.load_state_dic(torch.load(load_path+'/model.pth'), map_location=device)
            print("Model is loaded from {}".format(load_path+'/model.pth'))

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
        if not load_model and train_mode:
            if not os.path.exists(os.path.join(save_path, algorithm)):
                os.makedirs(save_path+algorithm)

            torch.save(self.model.state_dict(), save_path+algorithm+'/'+'%H-%M-%S-'+'model.pth')

            print("Save Model: {}".format(save_path+algorithm))

        elif load_model and train_mode:
            torch.save(self.model.state_dict(), save_path+algorithm+'/'+'%H-%M-%S-'+'model.pth')

            print("Save Model: {}".format(load_path))

    def train_model(self, state, action, reward, next_state, done):
        #print(state)
        state=torch.Tensor(state).to(device)
        next_state=torch.Tensor(next_state).to(device)

        one_hot_action=torch.zeros(2).to(device)
        one_hot_action[action]=1
        q=(self.model(state)*one_hot_action).sum()

        with torch.no_grad():
            max_Q=q.item()
            next_q=self.model(next_state)
            target_q=reward+next_q.max()*(discount_factor*(1-done))

        loss=F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), max_Q

if __name__=='__main__':
    model=DQN().to(device)
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)

    agent=DQNAgent(model, optimizer)

    model.train()

    step=0
    episode=0
    reward_list=[]
    loss_list=[]
    max_Q_list=[]

    while step<run_step+test_step:
        state=env.reset()

        episode_rewards=0
        done=False

        while not done:
            env.render()

            if step==run_step:
                train_mode=False
                model.eval()
            
            action=agent.get_action(state)
            next_state, reward, done, _=env.step(action)

            episode_rewards+=reward

            if train_mode==False:
                agent.epsilon=0.0

            if train_mode:
                if agent.epsilon>epsilon_min:
                    agent.epsilon-=1.0/run_step
                #print(agent.epsilon, 1.0/run_step)
                    
                loss, maxQ=agent.train_model(state, action, reward, next_state, done)
                loss_list.append(loss)
                max_Q_list.append(maxQ)

            if step%save_step==0 and step!=0 and train_mode:
                agent.save_model(load_model, train_mode)

            state=next_state
            step+=1

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

    agent.save_model(load_model, train_mode)
    env.close()