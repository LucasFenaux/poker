import os
# my computer has 14 cpus
# we keep half for my computer to be able to run things
# leaves us with 7
# 1 for leaderboard
# 4 tables
# 2 trainers
GAME_TYPES=["HOLDEM", "KUHN"]
# NUM_CPUS=os.environ["RAY_NUM_CPUS"]  # Set your desired CPU limit here
# GAME_TYPE="HOLDEM"
GAME_TYPE="KUHN"
NUM_CPUS=os.environ.get("RAY_NUM_CPUS",10)
NUM_GPUS=0
IS_RECURRENT=True


# local settings
# NUM_PLAYERS=10
# NUM_TABLES=3
# NUM_TRAINERS=3
# RESOURCE_LIMITED=True

# ripple settings
if GAME_TYPE=="HOLDEM":
    MAX_TABLE_SIZE=2
    NUM_PLAYERS=50
    NUM_TABLES=30
    NUM_TRAINERS=30
    RESOURCE_LIMITED=False
elif GAME_TYPE=="KUHN":
    MAX_TABLE_SIZE=2
    NUM_PLAYERS=50
    NUM_TABLES=30
    NUM_TRAINERS=30
    RESOURCE_LIMITED=False
else:
    raise NotImplementedError