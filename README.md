# V2: New distribution paradigm

Tables and Trainers.

We use docker to bypass the ray/mac M4 issue: https://github.com/ray-project/ray/issues/61495

## How to run
To start the container for PPO self-play training. Adjust the settings in `src/ppo_self_play/global_settings.py`, then:

`docker compose up --build`

To end the container

`docker compose down`

If you want to pretrain the models using behavior cloning. Use the docker-compose file in the `run/behavior_cloning/` folder 
to generate the data and the `run/behavior_cloning/train.py` script to pre-train the PPO actors. Make sure to properly specify
the model path in `run/ppo_self_play/main.py` to ensure it is using the pre-trained model.

## Logging
### Leaderboard
To display the live leaderboard, simply run in a separate shell

`uv run leaderboard_gui.py`

### Tensorboard
To view the tensorboard logs, run in a separate shell

`uv run tensorboard --logdir=./results --reload_multifile=true --reload_interval=5`

and open http://localhost:6006 on your browser

### Ray Dashboard

Open http://localhost:8265 on your browser

### Get results from the Docker container
The `get_from_docker` fetches the run results and models.

### Play with your Poker AI
The `run\app\run.py` script launches a small poker app UI to play poker with one of your pre-trained AIs.