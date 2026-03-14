# V2: New distribution paradigm

Tables and Trainers.

We use docker to bypass the ray/mac M4 issue: https://github.com/ray-project/ray/issues/61495

## How to run
To start the container

`docker compose up --build`

To end the container

`docker compose down`

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


[//]: # (`docker build -t poker-casino . ` )
[//]: # ()
[//]: # (`docker run -it --rm --shm-size="8g" -p 10001:10001 -p 6379:6379 -p 8265:8265 -v "$&#40;pwd&#41;/results:/app/results" poker-casino`)
