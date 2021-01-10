import random
import numpy as np
from typing import List, Dict, Optional, Tuple
import copy

from tensorforce import Agent

from site_location import (
    SiteLocationPlayer,
    Store,
    SiteLocationMap,
    euclidian_distances,
    attractiveness_allocation,
)


class TensorforcePlayer(SiteLocationPlayer):
    def __init__(self, player_id: int, config: Dict):
        super(TensorforcePlayer, self).__init__(player_id, config)

        agent = Agent.create(
            agent="tensorforce",
            update=64,
            optimizer=dict(optimizer="adam", learning_rate=1e-3),
            objective="action_value",
            reward_estimation=dict(horizon=20),
            states=dict(
                population_density=dict(type='float', shape=(400, 400, 1)),
                self_stores=dict(type='float', shape=(400, 400, 1)),
                other_stores=dict(type='float', shape=(400, 400, 4)),
            ),
            actions=dict(type='float', shape=(40, 40), min_value=0, max_value=1),
            policy=[
                [
                    dict(type='retrieve', tensors=['population_density']),
                    dict(type='conv2d', size=64),
                    dict(type='flatten'),
                    dict(type='register', tensor='obs-population_density')
                ],
                [
                    dict(type='retrieve', tensors=['self_stores']),
                    dict(type='conv2d', size=32),
                    dict(type='flatten'),
                    dict(type='register', tensor='obs-self_stores')
                ],
                [
                    dict(type='retrieve', tensors=['other_stores']),
                    dict(type='conv2d', size=32),
                    dict(type='flatten'),
                    dict(type='register', tensor='obs-other_stores')
                ],
                [
                    dict(
                        type='retrieve', aggregation='concat',
                        tensors=['obs-population_density', 'obs-self_stores', 'obs-other_stores']
                    ),
                    dict(type='dense', size=64)
                ]
            ],
        )

        self.agent = agent
        self.actions = None
        self.states = None
        self.internals = agent.initial_internals()
        self.reward = 0

    def get_experience(self):
        return self.actions, self.internals, self.states

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        store_conf = self.config["store_config"]

        temp_store_locations = copy.deepcopy(store_locations)
        alloc = attractiveness_allocation(
            slmap, temp_store_locations, store_conf
        )

        self.states = {
            "population_density": slmap.population_distribution,
            "self_stores": alloc.pop(self.player_id),
            "other_stores": np.array(list(alloc.values()))
        }

        self.reward = current_funds
        self.actions, self.internals = self.agent.act(
            states=self.states, internals=self.internals, independent=True
        )

        ind = np.unravel_index(np.argmax(self.actions, axis=None), self.actions.shape)

        x_coord = int(ind[0] * 10 + 5)
        y_coord = int(ind[1] * 10 + 5)

        if current_funds >= store_conf["large"]["capital_cost"]:
            store_type = "large"
        elif current_funds >= store_conf["medium"]["capital_cost"]:
            store_type = "medium"
        else:
            store_type = "small"

        store = Store(
            (
                x_coord,
                y_coord,
            ),
            store_type,
        )

        self.stores_to_place = [store]
