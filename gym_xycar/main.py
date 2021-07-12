import gym

env = gym.make('gym_xycar:xycar-v0')

for i_episode in range(20000):
    observation = env.reset()

    t=0
    while True:
        env.render()

        action = env.action_space.sample()
        #print('action', action)
        observation, reward, done, info = env.step(action)
        #print('state', observation)

        if done:
            print("Episode finished after {} timesteps".format(t+1))

            break

        t+=1

env.close()