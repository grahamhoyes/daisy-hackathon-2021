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


def player_attractiveness_allocation(
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
        best_attractiveness = player_attractiveness_allocation(
            stores[player_id], slmap, store_config
        )
        attractiveness_by_player[player_id] = best_attractiveness

    return attractiveness_by_player


def normalized_attractiveness_allocation(slmap, stores, attractiveness_by_player):
    total_attractiveness = np.zeros(slmap.size)
    for player_id in stores:
        total_attractiveness += attractiveness_by_player[player_id]
    total_attractiveness = np.where(total_attractiveness == 0, 1, total_attractiveness)

    player_allocations = {}
    for player_id in stores:
        allocation = attractiveness_by_player[player_id] / total_attractiveness
        player_allocations[player_id] = allocation

    return player_allocations


class GridStrideAllocPlayer(SiteLocationPlayer):
    """
    Agent samples locations and selects the highest allocating one using
    the allocation function.
    """

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        store_conf = self.config["store_config"]
        step_size = 20
        all_sample_pos = []

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

        best_score = 0
        best_pos = []

        attractiveness_by_player = all_players_attractiveness_allocation(
            slmap, store_locations, store_conf
        )

        for pos in sample_pos:
            sample_store = Store(pos, store_type)
            temp_store_locations = copy.deepcopy(store_locations)
            temp_store_locations[self.player_id].append(sample_store)

            temp_attractiveness_by_player = copy.deepcopy(attractiveness_by_player)

            sample_player_alloc = player_attractiveness_allocation(
                temp_store_locations[self.player_id], slmap, store_conf
            )
            temp_attractiveness_by_player[self.player_id] = sample_player_alloc
            sample_alloc = normalized_attractiveness_allocation(
                slmap, temp_store_locations, temp_attractiveness_by_player
            )
            sample_score = (
                sample_alloc[self.player_id] * slmap.population_distribution
            ).sum()
            if sample_score > best_score:
                best_score = sample_score
                best_pos = [pos]
            elif sample_score == best_score:
                best_pos.append(pos)

        # store_types = ["small", "medium", "large"]
        # best_scores = [0, 0, 0]
        # best_positions = [[], [], []]

        # for i in range(len(store_types)):
        #     store_type = store_types[i]
        #     if current_funds < store_conf[store_type]["capital_cost"]:
        #         continue
        #     for pos in sample_pos:
        #         sample_store = Store(pos, store_type)
        #         temp_store_locations = copy.deepcopy(store_locations)
        #         temp_store_locations[self.player_id].append(sample_store)
        #         sample_alloc = attractiveness_allocation(
        #             slmap, temp_store_locations, store_conf
        #         )
        #         sample_score = (
        #             sample_alloc[self.player_id] * slmap.population_distribution
        #         ).sum()
        #         if sample_score > best_scores[i]:
        #             best_scores[i] = sample_score
        #             best_positions[i] = [pos]
        #         elif sample_score == best_scores[i]:
        #             best_positions.append(pos)

        # best_index = np.argmax(best_scores)
        # best_pos = best_positions[best_index]
        # store_type = store_types[best_index]

        # import pdb; pdb.set_trace()
        # max_alloc_positons = np.argwhere(alloc[self.player_id] == np.amax(alloc[self.player_id]))
        # pos = random.choice(max_alloc_positons)
        self.stores_to_place = [Store(random.choice(best_pos), store_type)]
        return


class MaxDensityAllocPlayer(SiteLocationPlayer):
    """
    A mix of the max density and allocation agents

    - Choose the top locations according to density
    - Apply the allocation agent to these locations
    - Run separately for each store type than can be afforded,
      pick the best scoring move
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sorted_density_indices = None

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        store_conf = self.config["store_config"]

        # Check if any stores can be bought at all
        if current_funds < store_conf["small"]["capital_cost"]:
            self.stores_to_place = []
            return

        all_stores_pos = []
        for player, player_stores in store_locations.items():
            for player_store in player_stores:
                all_stores_pos.append(np.array(player_store.pos))

        # Sort store positions by highest population density
        if self.sorted_density_indices is None:
            self.sorted_density_indices = np.dstack(
                np.unravel_index(
                    np.argsort(slmap.population_distribution.ravel()), slmap.size
                )
            )[0][::-1]

        store_type = "small"
        for _store_type in ["medium", "large"]:
            if current_funds >= store_conf[_store_type]["capital_cost"]:
                store_type = _store_type
            else:
                break

        # Filter positions that are too close to other stores
        min_dist = store_conf[store_type]["attractiveness"]
        num_positions_to_consider = 100
        legal_indices = []

        if all_stores_pos:
            for pos in self.sorted_density_indices:
                too_close = False
                for nearby_pos in all_stores_pos:
                    dist = np.linalg.norm(pos - nearby_pos)
                    if dist < min_dist:
                        too_close = True
                        break
                if not too_close:
                    legal_indices.append(tuple(pos))
                    if len(legal_indices) == num_positions_to_consider:
                        break
        else:
            # Account for no stores
            legal_indices = list(map(tuple, self.sorted_density_indices))[
                :num_positions_to_consider
            ]

        attractiveness_by_player = all_players_attractiveness_allocation(
            slmap, store_locations, store_conf
        )

        best_score = 0.0
        self.stores_to_place = []

        for pos in legal_indices:
            sample_store = Store(pos, store_type)
            temp_store_locations = copy.deepcopy(store_locations)
            temp_store_locations[self.player_id].append(sample_store)

            temp_attractiveness_by_player = copy.deepcopy(attractiveness_by_player)

            sample_player_alloc = player_attractiveness_allocation(
                temp_store_locations[self.player_id], slmap, store_conf
            )
            temp_attractiveness_by_player[self.player_id] = sample_player_alloc
            sample_alloc = normalized_attractiveness_allocation(
                slmap, temp_store_locations, temp_attractiveness_by_player
            )
            sample_score = (
                sample_alloc[self.player_id] * slmap.population_distribution
            ).sum()

            if sample_score > best_score:
                best_score = sample_score
                self.stores_to_place = [sample_store]

        if not self.stores_to_place:
            # Place the most expensive store we can afford
            for store_type in {"small", "medium", "large"}:
                if current_funds <= store_conf[store_type]["capital_cost"]:
                    self.stores_to_place = [Store(legal_indices[0], store_type)]
                else:
                    break


class StorePotentialAllocationPlayer(SiteLocationPlayer):
    """
    potential = (1 - your attractiveness) * density

    Strategy:
    - Divide the grid into 20x20 sub-grids.
    - Compute the potential of the entire board for this player, average in each sub-grid to smooth.
    - Pick the block with the highest potential, pick somewhere in there to put it
        - Disqualify a block if you have a store in it already
    - This does not currently consider other players
    """

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        store_conf = self.config["store_config"]
        step_size = 20

        # Choose the largest store possible
        if current_funds >= store_conf["large"]["capital_cost"]:
            store_type = "large"
        elif current_funds >= store_conf["medium"]["capital_cost"]:
            store_type = "medium"
        elif current_funds >= store_conf["small"]["capital_cost"]:
            store_type = "small"
        else:
            self.stores_to_place = []
            return

        attractiveness = player_attractiveness_allocation(
            store_locations[self.player_id], slmap, store_conf
        )

        # attractiveness_by_player = all_players_attractiveness_allocation(slmap, store_locations, store_conf)
        # total_attractiveness = np.zeros(slmap.size)
        # for player_id in attractiveness_by_player:
        #     total_attractiveness += attractiveness_by_player[player_id]
        # # total_attractiveness = np.where(total_attractiveness == 0, 1, total_attractiveness)
        # attractiveness = total_attractiveness

        potential = (1 - attractiveness) * slmap.population_distribution
        # potential = slmap.population_distribution / attractiveness

        grid_size = (
            self.config["map_size"][0] // step_size,
            self.config["map_size"][1] // step_size,
        )

        potential_average = np.zeros(grid_size)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                sub_grid = potential[
                    i * step_size : (i + 1) * step_size,
                    j * step_size : (j + 1) * step_size,
                ]
                potential_average[i][j] = sub_grid.mean()

        sorted_potential_average_indices = np.dstack(
            np.unravel_index(
                np.argsort(-potential_average.ravel()),
                potential_average.shape,
            )
        )[0]

        # Find the first block that we don't have a store already in
        our_stores_pos = [
            np.array(store.pos) for store in store_locations[self.player_id]
        ]
        min_dist = 50

        for block_position in sorted_potential_average_indices:
            # Convert block positions back to map positions (at the center of each block)
            map_position = block_position * step_size + step_size // 2
            too_close = False

            for other_pos in our_stores_pos:
                dist = np.linalg.norm(other_pos - map_position)
                if dist < min_dist:
                    too_close = True
                    break

            if not too_close:
                self.stores_to_place = [
                    Store((map_position[0], map_position[1]), store_type)
                ]
                return
