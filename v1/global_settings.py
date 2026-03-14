import os

# CRITICAL MACOS FIX: Stop Apple's Accelerate framework from thread explosion
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

NUM_CPUS=os.cpu_count()  # leave some for my laptop
NUM_GPUS=0   # leave some for my laptop
NUM_PLAYERS=6
NUM_GAMES=100

PLAYER_CPU_LIMIT = 1
TABLE_CPU_LIMIT = 1
# PLAYER_CPU_LIMIT=max(1, NUM_CPUS// NUM_PLAYERS)
# TABLE_CPU_LIMIT=max(1, (NUM_CPUS - (PLAYER_CPU_LIMIT * NUM_PLAYERS) - 1)//(NUM_PLAYERS//2))  # we assume min_table_size=2


