import ray

@ray.remote(num_cpus=0.1)
class LeaderboardActor:
    def __init__(self, save_folder: str = "./"):
        self.save_folder = save_folder

    def update(self, player_winnings: dict[int: float]):
        pass

    def save(self):
        pass