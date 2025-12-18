import gym
import slimevolleygym
import random
import neat
import numpy
import visualize
from neat.parallel import ParallelEvaluator

def my_policy(obs, net):
  output = net.activate(obs)
  action = [
    1 if output[0] > 0.75 else 0,
    1 if output[1] > 0.75 else 0,
    1 if output[2] > 0.75 else 0,
  ]
  return action



def eval_genome(genome, config):

  N_EPISODES = 5
  net = neat.nn.FeedForwardNetwork.create(genome, config)
  total_fitness = 0

  for ep in range(N_EPISODES):


    env = gym.make("SlimeVolley-v0")
    obs = env.reset()
    done = False
    fitness = 0

    while not done:
      #env.render()
      action = my_policy(obs, net)
      obs, reward, done, info = env.step(action)
      fitness += reward
    
    total_fitness += fitness
  
  env.close()
  return total_fitness / N_EPISODES

def eval_genomes(genomes, config):
  for genome_id, genome in genomes:
    genome.fitness = eval_genome(genome, config)

config = neat.Config(
  neat.DefaultGenome,
  neat.DefaultReproduction,
  neat.DefaultSpeciesSet,
  neat.DefaultStagnation,
  "config-neat"
)

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.StatisticsReporter())

stats = neat.StatisticsReporter()
p.add_reporter(stats)

num_cores = 10  
pe = ParallelEvaluator(num_cores, eval_genome)
winner = p.run(pe.evaluate, 50)

# Test the winner
print('\nBest genome:\n{!s}'.format(winner))
print('\nBest genome ever: \n{}'.format(stats.best_genome()))

best_genome_ever = stats.best_genome()

net = neat.nn.FeedForwardNetwork.create(best_genome_ever, config)

# env = gym.make('SlimeVolley-v0')
# obs = env.reset()
# done = False
# total_reward = 0

# while not done:
#   env.render()
#   action = my_policy(obs, net)
#   obs, reward, done, info = env.step(action)
#   total_reward += reward

# env.close()
# print("The total reward is: ", total_reward)

print(net)

# visualize.plot_stats(stats, ylog = True, view = True)
# visualize.plot_species(stats, view = True)

# node_names = {-1: 'a', -2: 'b', -3: 'c', -4: 'd', -5: 'e', -6: 'f', -7:'g', -8:'h', -9:'i', -10:'j', -11:'k', -12:'l' , 0: '0', 1: '1', 2: '2'}
# visualize.draw_net(config, winner, view = True, node_names = node_names, prune_unused = True)
