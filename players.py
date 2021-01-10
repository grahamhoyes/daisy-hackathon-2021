import random
import numpy as np
from typing import List, Dict, Optional, Tuple
import copy

from site_location import (
    SiteLocationPlayer,
    Store,
    SiteLocationMap,
    euclidian_distances,
    attractiveness_allocation,
)


class FooPlayer(SiteLocationPlayer):
    pass


class GridStrideAllocPlayer(SiteLocationPlayer):
    """
    Agent samples locations and selects the highest allocating one using
    the allocation function.
    """
    def __init__(self, player_id: int, config: Dict):
        super().__init__(player_id, config)
        self.round = 0

    def player_attractiveness_allocation(
        self,
        stores,
        slmap,
        store_config,
    ):
        best_attractiveness = np.zeros(slmap.size)
        for store in stores:
            distances = euclidian_distances(slmap.size, store.pos)
            attractiveness = (
                store_config[store.store_type]["attractiveness"]
                / np.maximum(distances, np.ones(distances.shape))
                - store_config[store.store_type]["attractiveness_constant"]
            )
            attractiveness = np.where(attractiveness < 0, 0, attractiveness)
            best_attractiveness = np.maximum(best_attractiveness, attractiveness)

        return best_attractiveness

    def all_players_attractiveness_allocation(
        self,
        slmap: SiteLocationMap,
        stores: Dict[int, List[Store]],
        store_config: Dict[str, Dict[str, float]],
    ) -> Dict[int, np.ndarray]:
        """Returns population allocation per player for the given map, players and stores.

        Allocation for a given player is a numpy array of the same size as the map,
        with the fraction of the population allocated to that player in each grid
        location.

        Each grid location will be allocated to the players based on a ratio of
        attractiveness of the stores to that grid location.

        attractiveness = store_attractiveness / distance - store_attractiveness_constant

        For a given player, only the store with the max attractiveness to a given
        grid location is considered (ie. doubling up on stores in the same location
        will not result in more population).

        Arguments:
        - slmap: SiteLocationMap object
        - stores: all stores for each player by id
        - store_config: configuration from the game config
        """

        attractiveness_by_player = {}
        for player_id in stores:
            best_attractiveness = self.player_attractiveness_allocation(
                stores[player_id], slmap, store_config
            )
            attractiveness_by_player[player_id] = best_attractiveness

        return attractiveness_by_player

    def normalized_attractiveness_allocation(
        self, slmap, stores, attractiveness_by_player
    ):
        total_attractiveness = np.zeros(slmap.size)
        for player_id in stores:
            total_attractiveness += attractiveness_by_player[player_id]
        total_attractiveness = np.where(
            total_attractiveness == 0, 1, total_attractiveness
        )

        player_allocations = {}
        for player_id in stores:
            allocation = attractiveness_by_player[player_id] / total_attractiveness
            player_allocations[player_id] = allocation

        return player_allocations

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        store_conf = self.config["store_config"]
        step_size = 25
        all_sample_pos = []

        self.round += 1

        if self.round > 9:
            self.stores_to_place = []
            return 

        x_pos = np.arange(0, slmap.size[0], step_size) + int(step_size / 2)
        y_pos = np.arange(0, slmap.size[1], step_size) + int(step_size / 2)
        for i in range(len(x_pos)):
            x = x_pos[i]
            for j in range(len(y_pos)):
                y = y_pos[j]
                all_sample_pos.append((x, y))

        # sample_pos = random.sample(all_sample_pos, 200)
        sample_pos = all_sample_pos

        # Choose largest store type possible:
        if current_funds >= store_conf["large"]["capital_cost"]:
            store_type = "large"
        elif current_funds >= store_conf["medium"]["capital_cost"]:
            store_type = "medium"
        else:
            store_type = "small"

        attractiveness_by_player = self.all_players_attractiveness_allocation(
            slmap, store_locations, store_conf
        )

        # store_types = ["small", "medium", "large"]
        # best_scores = [0, 0, 0]
        # best_positions = [[], [], []]


        def find_store_placement(
            store_types,
            best_scores, 
            best_positions,
            store_locations
        ):
            attractiveness_by_player = self.all_players_attractiveness_allocation(
                slmap, store_locations, store_conf
            )
            for i in range(len(store_types)):
                store_type = store_types[i]
                if current_funds < store_conf[store_type]["capital_cost"]:
                    continue
                for pos in sample_pos:
                    sample_store = Store(pos, store_type)
                    temp_store_locations = copy.deepcopy(store_locations)
                    temp_store_locations[self.player_id].append(sample_store)

                    temp_attractiveness_by_player = copy.deepcopy(attractiveness_by_player)

                    sample_player_alloc = self.player_attractiveness_allocation(
                        temp_store_locations[self.player_id], slmap, store_conf
                    )
                    temp_attractiveness_by_player[self.player_id] = sample_player_alloc
                    sample_alloc = self.normalized_attractiveness_allocation(
                        slmap, temp_store_locations, temp_attractiveness_by_player
                    )
                    sample_score = (
                        sample_alloc[self.player_id] * slmap.population_distribution
                    ).sum()
                    if sample_score > best_scores[i]:
                        best_scores[i] = sample_score
                        best_positions[i] = [pos]
                    elif sample_score == best_scores[i]:
                        best_positions[i].append(pos)

                # break 

            
        store_types = ["large", "medium", "small"]
        best_scores = [0, 0, 0]
        best_positions = [[], [], []]


        self.stores_to_place = []

        find_store_placement(
            store_types, best_scores, best_positions, store_locations
        )

        best_index = np.argmax(best_scores)
        best_pos = best_positions[best_index]
        store_type = store_types[best_index]

        self.stores_to_place.append(Store(random.choice(best_pos), store_type))

        sample_store = Store(random.choice(best_pos), store_type)
        temp_store_locations = copy.deepcopy(store_locations)
        temp_store_locations[self.player_id].append(sample_store)


        # store_types = ["small", "medium", "large"]
        # best_scores = [0, 0, 0]
        # best_positions = [[], [], []]

        # find_store_placement(
        #     store_types, best_scores, best_positions, temp_store_locations
        # )

        # best_index = np.argmax(best_scores)
        # best_pos = best_positions[best_index]
        # store_type = store_types[best_index]

        # self.stores_to_place.append(Store(random.choice(best_pos), store_type))
        return
