

class BaseDataStorage:
    def __init__(self, *args, **kwargs):
        pass

    def add(self, player_id, hand_info_ref, num_samples, *args, **kwargs):
        raise NotImplementedError

    def can_train(self, player_id):
        raise NotImplementedError

    def get_batch(self, player_id):
        raise NotImplementedError


class DataStorage(BaseDataStorage):
    # store single transitions that are treated as independent
    def __init__(self, player_ids, batch_size: int, on_policy):
        super(DataStorage, self).__init__()
        self.player_ids = player_ids
        self.batch_size = batch_size
        self.num_samples = {player_id: 0 for player_id in player_ids}
        self.samples = {player_id: [] for player_id in player_ids}
        self.on_policy = on_policy

    def add(self, player_id, hand_info_ref, num_samples, *args, **kwargs):
        self.num_samples[player_id] += num_samples
        self.samples[player_id].append(hand_info_ref)
        return self.can_train(player_id)

    def can_train(self, player_id):
        if self.num_samples[player_id] >= self.batch_size:
            return True  # can train

        return False

    def get_batch(self, player_id):
        assert self.num_samples[player_id] >= self.batch_size
        samples = self.samples[player_id]
        self.samples[player_id] = []
        self.num_samples[player_id] = 0
        # let the trainer handle extra data
        return samples, self.num_samples[player_id]
