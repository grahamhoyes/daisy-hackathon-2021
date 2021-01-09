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


class TensorforcePlayer(SiteLocationPlayer):

    def __init__(self, player_id: int, config: Dict, agent):
        super().__init__(player_id, config)
        self.agent = agent
        self.actions = None
        self.internals = agent.initial_internals()

    def get_actions_and_internals(self):
        return self.actions, self.internals

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        actions, self.internals = self.agent.act(
            states=states, internals=self.internals, independent=True
        )

        # self_stores_pos = []
        # for store in store_locations[self.player_id]:
        #     self_stores_pos.append(store.pos)
        #
        # opp_store_locations = {
        #     k: v for (k, v) in store_locations.items() if k != self.player_id
        # }
        # opp_all_stores = []
        # for player, player_stores in opp_store_locations.items():
        #     for player_store in player_stores:
        #         if player_store.pos not in self_stores_pos:
        #             opp_all_stores.append(player_store)
        # if not opp_all_stores:
        #     self.stores_to_place = []
        #     return
        # else:
        #     self.stores_to_place = [random.choice(opp_all_stores)]
        #     return