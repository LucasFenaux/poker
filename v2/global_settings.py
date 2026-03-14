import os
# my computer has 14 cpus
# we keep half for my computer to be able to run things
# leaves us with 7
# 1 for leaderboard
# 4 tables
# 2 trainers

NUM_CPUS=os.cpu_count()//2  # leave some for my laptop = 7
NUM_GPUS=0   # leave some for my laptop
NUM_PLAYERS=6
NUM_TABLES= 1
NUM_TRAINERS=1



