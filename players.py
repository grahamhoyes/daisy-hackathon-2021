import random
import numpy as np
from typing import List, Dict, Optional, Tuple
import copy
import time

from site_location import (
    SiteLocationPlayer,
    Store,
    SiteLocationMap,
    euclidian_distances,
    attractiveness_allocation,
)


class FooPlayer(SiteLocationPlayer):
    pass


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
        # TODO: Make this more efficient
        start = time.time()
        if self.sorted_density_indices is None:
            self.sorted_density_indices = np.dstack(
                np.unravel_index(
                    np.argsort(slmap.population_distribution.ravel()), slmap.size
                )
            )[0][::-1]

        # Filter positions that are too close to other stores
        # TODO: Vary min_dist with store type
        min_dist = 50
        num_positions_to_consider = 400
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
        end = time.time()

        print(f"Sorting runtime: {end - start}")

        best_score = 0.0
        best_store = None

        start = time.time()
        for store_type in {"large", "medium", "small"}:
            if current_funds < store_conf[store_type]["capital_cost"]:
                continue

            for pos in legal_indices:
                sample_store = Store(pos, store_type)
                temp_store_locations = copy.deepcopy(store_locations)
                temp_store_locations[self.player_id].append(sample_store)
                sample_alloc = attractiveness_allocation(
                    slmap, temp_store_locations, store_conf
                )
                sample_score = (
                    sample_alloc[self.player_id] * slmap.population_distribution
                ).sum()

                if sample_score > best_score:
                    best_score = sample_score
                    best_store = sample_store

        end = time.time()
        print(f"Store selection runtime: {end - start}")

        if not best_store:
            # Place the most expensive store we can afford
            best_store = None
            for store_type in {"small", "medium", "large"}:
                if current_funds <= store_conf[store_type]["capital_cost"]:
                    best_store = Store(legal_indices[0], store_type)
                else:
                    break

        print(f"Attempting to place a {best_store.store_type} at {best_store.pos}")
        self.stores_to_place = [best_store] if best_store else []
