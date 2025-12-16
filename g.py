import gym
import slimevolleygym

env = gym.make("SlimeVolley-v0")

obs = env.reset()
done = False
total_reward = 0

while not done:
  action = my_policy(obs)
  obs, reward, done, info = env.step(action)
  total_reward += reward
  env.render()

print("score:", total_reward)