import random
import itertools
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
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


def calculate_total_attractiveness(slmap, stores, attractiveness_by_player):
    total_attractiveness = np.zeros(slmap.size)
    for player_id in stores:
        total_attractiveness += attractiveness_by_player[player_id]
    total_attractiveness = np.where(total_attractiveness == 0, 1, total_attractiveness)
    return total_attractiveness


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


def compute_sample_score(
    store: Store,
    player_id: int,
    slmap: SiteLocationMap,
    store_locations: Dict[int, List[Store]],
    store_config: Dict[str, Dict[str, float]],
):
    """
    Compute the sample score obtained by adding store ``store`` for the current
    player
    """

    attractiveness_by_player = all_players_attractiveness_allocation(
        slmap, store_locations, store_config
    )

    temp_store_locations = copy.deepcopy(store_locations)

    if store is not None:
        temp_store_locations[player_id].append(store)

    temp_attractiveness_by_player = copy.deepcopy(attractiveness_by_player)

    sample_player_alloc = player_attractiveness_allocation(
        temp_store_locations[player_id], slmap, store_config
    )
    temp_attractiveness_by_player[player_id] = sample_player_alloc
    sample_alloc = normalized_attractiveness_allocation(
        slmap, temp_store_locations, temp_attractiveness_by_player
    )
    sample_score = (sample_alloc[player_id] * slmap.population_distribution).sum()

    return sample_score


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

        self.stores_to_place = [Store(random.choice(best_pos), store_type)]
        return


class NoisyMolecularWhalesPlayer(SiteLocationPlayer):
    """
    Agent samples locations and selects the highest allocating one using
    the allocation function.
    """

    def __init__(self, player_id: int, config: Dict):
        super().__init__(player_id, config)
        self.round = 0

    # helper function to compute the change in allocation based on new store placements
    def find_store_placement(
        self,
        store_types,
        sample_pos,
        current_funds,
        slmap,
        store_locations,
        config,
        player_id,
    ):
        best_scores = []
        best_positions = []
        for i in range(len(store_types)):
            best_scores.append([])
            best_positions.append([])

        attractiveness_by_player = all_players_attractiveness_allocation(
            slmap, store_locations, config["store_config"]
        )

        current_alloc = normalized_attractiveness_allocation(
            slmap, store_locations, attractiveness_by_player
        )
        current_score = (
            current_alloc[player_id] * slmap.population_distribution
        ).sum() * config["profit_per_customer"]

        for i in range(len(store_types)):
            store_type = store_types[i]
            store_cost = config["store_config"][store_type]["capital_cost"]
            store_maintenance = config["store_config"][store_type]["operating_cost"]

            if current_funds < store_cost:
                continue
            for pos in sample_pos:
                sample_store = Store(pos, store_type)
                temp_store_locations = copy.deepcopy(store_locations)
                temp_store_locations[player_id].append(sample_store)

                temp_attractiveness_by_player = copy.deepcopy(attractiveness_by_player)

                sample_player_alloc = player_attractiveness_allocation(
                    temp_store_locations[player_id], slmap, config["store_config"]
                )
                temp_attractiveness_by_player[player_id] = sample_player_alloc
                sample_alloc = normalized_attractiveness_allocation(
                    slmap, temp_store_locations, temp_attractiveness_by_player
                )
                sample_score = (
                    sample_alloc[player_id] * slmap.population_distribution
                ).sum() * config["profit_per_customer"]

                delta_score = sample_score - current_score
                if delta_score > store_cost * 0.33:
                    best_scores[i].append(sample_score)
                    best_positions[i].append(pos)

        return best_scores, best_positions

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        store_conf = self.config["store_config"]
        step_size = 20
        all_sample_pos = []

        # Partitions into grid and only takes boxes with population density over the mean + std
        half_window = int(step_size / 2)
        x_pos = np.arange(0, slmap.size[0], step_size) + half_window
        y_pos = np.arange(0, slmap.size[1], step_size) + half_window
        low_density_count = 0
        for i in range(len(x_pos)):
            x = x_pos[i]
            for j in range(len(y_pos)):
                y = y_pos[j]

                x += np.random.randint(-2, 2 + 1)
                y += np.random.randint(-2, 2 + 1)
                population_density = slmap.population_distribution[
                    x - half_window : x + half_window, y - half_window : y + half_window
                ]
                if (
                    population_density.mean()
                    < slmap.population_distribution.mean()
                    + slmap.population_distribution.std()
                ):
                    low_density_count += 1
                    continue

                all_sample_pos.append((x, y))

        # print("Grid Count:", len(x_pos) * len(y_pos) - low_density_count)

        # sample_pos = random.sample(all_sample_pos, 200)
        sample_pos = all_sample_pos

        self.stores_to_place = []
        temp_store_locations = copy.deepcopy(store_locations)
        for i in range(2):
            store_types = ["small", "medium", "large"]
            all_best_scores, all_best_positions = self.find_store_placement(
                store_types,
                sample_pos,
                current_funds,
                slmap,
                temp_store_locations,
                self.config,
                self.player_id,
            )

            best_scores = []
            best_positions = []
            for i in range(len(all_best_scores)):
                if len(all_best_scores[i]) > 0:
                    best_score = np.array(all_best_scores[i])

                    acceptable_scores = []
                    positions = []
                    threshold = best_score.max() * 0.90
                    for j in range(len(best_score)):
                        if best_score[j] > threshold:
                            acceptable_scores.append(best_score[j])
                            positions.append(all_best_positions[i][j])
                    index = np.random.randint(len(acceptable_scores))
                    best_scores.append(acceptable_scores[index])
                    best_positions.append(positions[index])

            try:
                # import pdb; pdb.set_trace()
                best_index = np.argmax(best_scores)
                best_pos = [best_positions[best_index]]
                store_type = store_types[best_index]
                self.stores_to_place.append(Store(random.choice(best_pos), store_type))
            except:
                return

            current_funds -= store_conf[store_type]["capital_cost"]
            sample_store = Store(random.choice(best_pos), store_type)
            temp_store_locations[self.player_id].append(sample_store)

        return


class MarginalCostPartitionPlayer(SiteLocationPlayer):
    """
    Agent samples locations and selects the highest allocating one using
    the allocation function.
    """

    def __init__(self, player_id: int, config: Dict):
        super().__init__(player_id, config)
        self.round = 0

    def find_store_placement(
        self,
        store_type,
        sample_pos,
        current_funds,
        slmap,
        store_locations,
    ):

        store_cost = self.config["store_config"][store_type]["capital_cost"]
        store_maintenance = self.config["store_config"][store_type]["operating_cost"]

        if current_funds < store_cost:
            return -np.inf, None

        attractiveness_by_player = all_players_attractiveness_allocation(
            slmap, store_locations, self.config["store_config"]
        )

        current_alloc = normalized_attractiveness_allocation(
            slmap, store_locations, attractiveness_by_player
        )
        current_score = (
            current_alloc[self.player_id] * slmap.population_distribution
        ).sum() * self.config["profit_per_customer"]

        best_score = 0.0
        best_store = None

        for pos in sample_pos:
            sample_store = Store(pos, store_type)
            temp_store_locations = copy.deepcopy(store_locations)
            temp_store_locations[self.player_id].append(sample_store)

            sample_player_alloc = player_attractiveness_allocation(
                temp_store_locations[self.player_id],
                slmap,
                self.config["store_config"],
            )
            attractiveness_by_player[self.player_id] = sample_player_alloc

            total_attractiveness = calculate_total_attractiveness(
                slmap, temp_store_locations, attractiveness_by_player
            )

            sample_alloc = (
                attractiveness_by_player[self.player_id] / total_attractiveness
            )

            sample_score = (
                sample_alloc * slmap.population_distribution
            ).sum() * self.config["profit_per_customer"]

            delta_score = sample_score - current_score
            if delta_score > store_cost * 0.33:
                if sample_score > best_score:
                    best_score = sample_score
                    best_store = sample_store

        delta_score = best_score - current_score
        print("Delta Score for {}: {:.0f}".format(store_type, delta_score))

        return best_score, best_store

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        store_conf = self.config["store_config"]
        step_size = 20
        all_sample_pos = []

        # Partitions into grid and only takes boxes with population density over the mean + std
        half_window = int(step_size / 2)
        x_pos = np.arange(0, slmap.size[0], step_size) + half_window
        y_pos = np.arange(0, slmap.size[1], step_size) + half_window
        threshold = (
            slmap.population_distribution.mean() + slmap.population_distribution.std()
        )
        low_density_count = 0
        for i in range(len(x_pos)):
            x = x_pos[i]
            for j in range(len(y_pos)):
                y = y_pos[j]

                x += np.random.randint(-1, 1 + 1)
                y += np.random.randint(-1, 1 + 1)
                population_density = slmap.population_distribution[
                    x - half_window : x + half_window, y - half_window : y + half_window
                ]
                mean_density = population_density.mean()

                if mean_density < threshold:
                    low_density_count += 1
                    continue

                all_sample_pos.append(((x, y), mean_density))

        # print("Grid Count:", len(x_pos) * len(y_pos) - low_density_count)

        # Sort the sample positions by decreasing density, so we cover them
        # before timing out
        all_sample_pos = sorted(all_sample_pos, key=lambda e: e[1], reverse=True)
        # Remove density from the array
        all_sample_pos = list(map(lambda e: e[0], all_sample_pos))

        # sample_pos = random.sample(all_sample_pos, 200)
        sample_pos = all_sample_pos

        self.stores_to_place = []
        temp_store_locations = copy.deepcopy(store_locations)
        for i in range(2):
            best_score = 0.0
            best_store = None

            for store_type in ["small", "medium", "large"]:
                score, store = self.find_store_placement(
                    store_type, sample_pos, current_funds, slmap, temp_store_locations
                )

                if score > best_score:
                    best_score = score
                    best_store = store

                    if i == 0:
                        # Make sure we return something before timing out
                        self.stores_to_place = [best_store]
                    else:
                        self.stores_to_place = [self.stores_to_place[0], best_store]

            if best_store is None:
                # If we couldn't place the first store, we won't be able to
                # place the second
                return

            # self.stores_to_place.append(best_store)

            current_funds -= store_conf[best_store.store_type]["capital_cost"]
            temp_store_locations[self.player_id].append(best_store)

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

    Things to try:
    - Change the stride size
    - Try placing multiple stores, see what works best
    - See if the potential gain (* number of remaining rounds) is > expected cost
    """

    def find_store_placement(
        self,
        store_type: str,
        step_size: int,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
    ):
        store_conf = self.config["store_config"]

        attractiveness = player_attractiveness_allocation(
            store_locations[self.player_id], slmap, store_conf
        )

        potential = (1 - attractiveness / 100) * slmap.population_distribution

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
            (store.store_type, np.array(store.pos))
            for store in store_locations[self.player_id]
        ]

        min_dists = {}
        store_sizes = ["small", "medium", "large"]
        for size1, size2 in itertools.product(store_sizes, store_sizes):
            min_dists[(size1, size2)] = (
                store_conf[size1]["attractiveness"]
                + store_conf[size2]["attractiveness"]
            ) / 2

        for block_position in sorted_potential_average_indices:
            # Convert block positions back to map positions (at the center of each block)
            map_position = block_position * step_size + step_size // 2
            too_close = False

            for other_store_type, other_pos in our_stores_pos:
                dist = np.linalg.norm(other_pos - map_position)
                # Allow some overlap of radaii
                if dist < min_dists[(store_type, other_store_type)] * 0.7:
                    too_close = True
                    break

            if not too_close:
                return Store((map_position[0], map_position[1]), store_type)

        return None

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        store_conf = self.config["store_config"]
        step_size = 10

        # Try the following combinations (if we can afford them),
        # see which gives the highest score:
        # - One large
        # - One large, one medium
        # - One large, one small
        # - Two medium
        # - One medium, one small
        # - Two small
        # - One medium
        # - One small

        def _test_two_stores(size1, size2):
            _store1 = self.find_store_placement(
                size1, step_size, slmap, store_locations
            )

            if _store1 is None:
                return -np.inf, None, None

            temp_store_locations = copy.deepcopy(store_locations)
            temp_store_locations[self.player_id].append(_store1)

            _store2 = self.find_store_placement(
                size2, step_size, slmap, temp_store_locations
            )

            if _store2 is None:
                return -np.inf, None, None

            _score = compute_sample_score(
                _store2, self.player_id, slmap, temp_store_locations, store_conf
            )
            return _score, _store1, _store2

        best_score = 0.0
        self.stores_to_place = []

        # One store
        for size in ["small", "medium", "large"]:
            if current_funds >= store_conf[size]["capital_cost"]:
                store = self.find_store_placement(
                    size, step_size, slmap, store_locations
                )
                score = compute_sample_score(
                    store, self.player_id, slmap, store_locations, store_conf
                )

                if score > best_score and store is not None:
                    self.stores_to_place = [store]
                    best_score = score

        # Two stores
        store_sizes = ["small", "medium", "large"]
        for size1, size2 in itertools.product(store_sizes, store_sizes):
            if (
                current_funds
                < store_conf[size1]["capital_cost"] + store_conf[size2]["capital_cost"]
            ):
                continue

            score, store1, store2 = _test_two_stores(size1, size2)

            if score > best_score and store1 is not None and store2 is not None:
                self.stores_to_place = [store1, store2]
                best_score = score
