import cProfile
import pstats
import sys
import train_rl

def run():
    train_rl.train_rl()

cProfile.run('run()', 'restats')
p = pstats.Stats('restats')
p.sort_stats('cumtime').print_stats(30)
