import random
import math
import numpy as np

from src.global_settings import (MAX_TABLE_SIZE, USE_HISTORICAL_SAMPLING,
                                 HISTORICAL_SAMPLING_RATE)


class JITTableScheduler:
    def __init__(self, table_min_size: int, table_max_size: int, player_ids, historical_sampling_receive_queue):
        self.player_ids = player_ids
        self.table_max_size = table_max_size
        self.table_min_size = table_min_size
        assert table_max_size <= MAX_TABLE_SIZE
        assert self.table_min_size <= self.table_max_size
        # self.max_plans = 10
        self.min_pool = max(table_max_size * 2, int(len(self.player_ids)/10))
        self.weights = {
            player_id: {
                other_player_id: 0 for other_player_id in player_ids if player_id != other_player_id
            } for player_id in player_ids
        }
        self.pool = set(self.player_ids[:])
        self.historical_sampling_receive_queue = historical_sampling_receive_queue
        self.historical_players_used = 0

    def update_weights(self, player_id: int, other_players: list[tuple[int, int]]):
        if other_players is not None:
            player_weights = self.weights[player_id]
            for other_player, other_player_version in other_players:
                if other_player != player_id and other_player >= 0:
                    player_weights[other_player] += other_player_version

    def add(self, player_id: int):
        """
        Add a player into the scheduler to be scheduled for a game
        :param player_id: Which player we are adding back in.
        :param other_players:  The other players the player was playing with if he was a table and their current version
        :return:
        """
        assert player_id not in self.pool, (f"Player duplicated - {player_id} is already in the pool")
        self.pool.add(player_id)

    def _get_history_table(self):
        # we first check if there are any historical checkpoints to assign
        if self.historical_sampling_receive_queue.qsize() < self.table_max_size - 1:
            # default back to the default get table
            return self._get_table()

        table = []

        available_players = list(self.pool)
        assert len(available_players) >= 1
        table_size = random.randint(self.table_min_size, self.table_max_size)

        starter_idx = random.randint(0, len(available_players) - 1)
        starter = available_players.pop(starter_idx)
        table.append(starter)

        for _ in range(table_size - 1):
            self.historical_players_used += 1
            table.append(-self.historical_players_used)  # -id indicates that the table should sample a historical player

        for player in table:
            if player >= 0:
                self.pool.remove(player)

        return table

    def _get_table(self):

        table = []

        available_players = list(self.pool)
        assert len(available_players) >= self.table_max_size
        table_size = random.randint(self.table_min_size, self.table_max_size)

        starter_idx = random.randint(0, len(available_players) - 1)
        starter = available_players.pop(starter_idx)
        table.append(starter)

        starter_weights = self.weights[starter]
        available_player_weights = []
        # we then get the weights of the starter and pick from the remaining players based on those weights
        for remaining_player in available_players:
            available_player_weights.append(starter_weights[remaining_player])

        # we invert them
        max_weight = max(available_player_weights)
        inverted_weights = [max_weight - remaining_player_weight + 1 for remaining_player_weight in available_player_weights]

        # then we normalize them
        normalized_player_weights = [inverted_weight / sum(inverted_weights) for
                                    inverted_weight in inverted_weights]

        assert math.isclose(sum(normalized_player_weights), 1.0, rel_tol=1e-5)

        # select the followers based on their weight
        followers = np.random.choice(available_players, p=normalized_player_weights, replace=False, size=table_size - 1)

        for follower in followers:
            follower = follower.item()
            available_players.remove(follower)
            table.append(follower)

        for player in table:
            self.pool.remove(player)

        return table

    def get_table(self):
        if len(self.pool) <= self.min_pool:
            return None
        if USE_HISTORICAL_SAMPLING:
            if random.random() < HISTORICAL_SAMPLING_RATE:
                return self._get_history_table()

        return self._get_table()


# obsolete
class PlanTableScheduler:
    def __init__(self, table_min_size: int, table_max_size: int, player_ids):
        self.player_ids = player_ids
        self.table_max_size = table_max_size
        self.table_min_size = table_min_size
        assert table_max_size <= MAX_TABLE_SIZE
        assert self.table_min_size <= self.table_max_size
        # self.max_plans = 10
        self.max_plans = np.inf
        self.min_pool = max(table_max_size * 2, int(len(self.player_ids)/10))
        self.plan_count = 0
        self.weights = {
            player_id: {
                other_player_id: 0 for other_player_id in player_ids if player_id != other_player_id
            } for player_id in player_ids
        }
        self.pool = set(self.player_ids[:])
        self.plans: list[list[set]] = []


    def _generate_plan(self):
        """
        We generate a partition of the all the players into tables based on their mutual weights at time of generation.
        :return: a plan, which is a list of disjoint sets such that the union of all those sets is self.player_ids
        """
        available_players = self.player_ids[:]
        plan = []
        while len(available_players) >= self.table_min_size:  # we accept that some players might not get to play a plan
            table = []
            # we pick a random table size
            if len(available_players) < self.table_max_size:
                table_size = len(available_players)
            else:
                table_size = random.randint(self.table_min_size, self.table_max_size)
                if len(available_players) - table_size < self.table_min_size:
                    table_size = self.table_min_size

            # we then pick a random player as the starter of the table
            starter_idx = random.randint(0, len(available_players)-1)
            starter = available_players.pop(starter_idx)
            table.append(starter)

            starter_weights = self.weights[starter]
            available_player_weights = []
            # we then get the weights of the starter and pick from the remaining players based on those weights
            for remaining_player in available_players:
                available_player_weights.append(starter_weights[remaining_player])

            # we invert them
            max_weight = max(available_player_weights)
            inverted_weights = [max_weight - remaining_player_weight + 1 for remaining_player_weight in available_player_weights]

            # then we normalize them
            normalized_player_weights = [inverted_weight / sum(inverted_weights) for
                                        inverted_weight in inverted_weights]

            assert math.isclose(sum(normalized_player_weights), 1.0, rel_tol=1e-5)

            # select the followers based on their weight
            followers = np.random.choice(available_players, p=normalized_player_weights, replace=False, size=table_size-1)

            for follower in followers:
                follower = follower.item()
                available_players.remove(follower)
                table.append(follower)

            plan.append(set(table))
        return plan

    def update_weights(self, player_id: int, other_players: list[tuple[int, int]]):
        if other_players is not None:
            player_weights = self.weights[player_id]
            for other_player, other_player_version in other_players:
                if other_player != player_id:
                    player_weights[other_player] += other_player_version

    def add(self, player_id: int):
        """
        Add a player into the scheduler to be scheduled for a game
        :param player_id: Which player we are adding back in.
        :param other_players:  The other players the player was playing with if he was a table and their current version
        :return:
        """
        assert player_id not in self.pool, (f"Player duplicated - {player_id} is already in the pool")
        self.pool.add(player_id)
        # self.was_updated = True

    def _find_table(self):
        # we get the current plan and check if a table is available with the players in the pool
        for i, plan in enumerate(self.plans):
            for j, potential_table in enumerate(plan):
                potential_table: set
                # we check if it is a possible table
                if potential_table.issubset(self.pool):
                    # we found a suitable table
                    table = plan.pop(j)
                    if len(plan) == 0:
                        self.plans.pop(i)  # we remove the empty list
                    # we update the pool
                    for player in table:
                        self.pool.remove(player)

                    return list(table)
        return None

    def get_table(self):
        if len(self.pool) < self.min_pool:
        # if not self.was_updated and self.already_returned_none:
            # we have not changed since last query and we already verified that no table is available, we lazily return
            return None

        # we get the current plan and check if a table is available with the players in the pool
        table = self._find_table()

        if table is not None:
            return table

        # if we got here, it means no suitable table was found
        # we first check if we already have too many plans
        if len(self.plans) >= self.max_plans:
            # too many plans, can't generate a new one, we wait until players catch up to move forward
            # self.already_returned_none = True
            return None

        # we still have room to create a new plan
        new_plan = self._generate_plan()
        print(f"Newest plan ({self.plan_count}): {new_plan} | {len(self.plans)}")
        self.plan_count += 1
        self.plans.append(new_plan)
        # self.was_updated = True

        # we try to find a table again
        table = self._find_table()
        return table