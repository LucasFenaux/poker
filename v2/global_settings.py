import os
# my computer has 14 cpus
# we keep half for my computer to be able to run things
# leaves us with 7
# 1 for leaderboard
# 4 tables
# 2 trainers

NUM_CPUS=os.environ["RAY_NUM_CPUS"]  # Set your desired CPU limit here
NUM_GPUS=0
NUM_PLAYERS=10
NUM_TABLES=3
NUM_TRAINERS=1



