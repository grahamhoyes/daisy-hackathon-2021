#!/usr/bin/env python3

import logging

logging.basicConfig()
log = logging.getLogger("site_location")
log.setLevel(logging.DEBUG)

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import random
from PIL import Image, ImageDraw  # type: ignore
import os
import shutil
import argparse
import importlib
import signal
import time

from copy import copy, deepcopy
from enum import Enum

from perlin_numpy import generate_perlin_noise_2d

from typing import List, Dict, Optional, Tuple


class Store:
    """
    Represents a placed store on the grid

    Note that store_type should be a string that matches the stores defined
    in the game configuration.
    """

    def __init__(self, pos: Tuple[int, int], store_type: str):
        self.pos = pos
        self.store_type = store_type


def blend_rgba(datas):
    """Return a numpy array that has blended RGBA data in the given list of
    numpy arrays
    """
    blended = np.zeros(datas[0].shape).astype(float)

    total_alpha = np.zeros(datas[0].shape[:2])
    np.where(total_alpha == 0, 1, total_alpha)

    for data in datas:
        data = data.astype(float) / 255.0
        total_alpha += data[:, :, 3]
        total_alpha = np.where(total_alpha == 0, 1, total_alpha)

    for data in datas:
        data = data.astype(float) / 255.0
        blended[:, :, :3] += (
            np.square(data[:, :, :3])
            * data[:, :, 3][:, :, None]
            / total_alpha[:, :, None]
        ) / len(datas)
        blended[:, :, 3] += data[:, :, 3] / len(datas)

    blended[:, :, :3] = np.sqrt(blended[:, :, :3])

    return (blended * 255.0).astype(np.uint8)


class SiteLocationMap:
    """
    Represent the site location game map.
    """

    def __init__(self, size, seed=0, population=1000000):
        self.size = size
        self.population = population

        noise = generate_perlin_noise_2d(size, (4, 4), (False, False))
        noise = np.where(noise < 0, 0, noise)
        noise *= population / np.sum(noise)

        self.population_distribution = noise

    def save_image(self, filename, players={}, stores={}, allocations={}):
        """Save an image of the map

        Arguments:
        - filename: desired output filename
        - players: Dict of SiteLocationPlayer objects by id
        - stores: stores for each player, by id
        - allocations: allocation percentages over the grid for each player, by id
        """
        data = np.zeros(
            (
                self.population_distribution.shape[0],
                self.population_distribution.shape[1],
                4,
            ),
            dtype=np.uint8,
        )

        pop_norm = self.population_distribution
        self.population_distribution *= 255.0 / pop_norm.max()

        data[:, :, 0] += pop_norm.astype(np.uint8)
        data[:, :, 1] += pop_norm.astype(np.uint8)
        data[:, :, 2] += pop_norm.astype(np.uint8)
        data[:, :, 3] = 255

        image = Image.fromarray(data, "RGBA")

        draw = ImageDraw.Draw(image)
        allocation_images = []
        for player_id, player in players.items():
            for store in stores[player_id]:
                pointsize = 3
                draw.ellipse(
                    (
                        store.pos[1] - pointsize,
                        store.pos[0] - pointsize,
                        store.pos[1] + pointsize,
                        store.pos[0] + pointsize,
                    ),
                    player.color,
                )

            allocation = allocations[player_id]
            al_data = np.zeros(
                (
                    self.population_distribution.shape[0],
                    self.population_distribution.shape[1],
                    4,
                ),
                dtype=np.uint8,
            )

            al_data[:, :, 0] = player.color[0]
            al_data[:, :, 1] = player.color[1]
            al_data[:, :, 2] = player.color[2]
            al_data[:, :, 3] = 255
            al_data[:, :, 3] = (
                allocation[:, :] * al_data[:, :, 3].astype(float)
            ).astype(np.uint8)
            allocation_images.append(al_data)

        blended = blend_rgba(allocation_images)
        # blended[:,:,:3] = np.where(pop_norm[:,:,None] == 0, 0, blended[:,:,:3])
        blended[:, :, 3] = pop_norm[:, :] * 2 / 255.0 * blended[:, :, 3]

        image.paste(Image.fromarray(blended), (0, 0), Image.fromarray(blended))
        image.save(filename)


class SiteLocationPlayer:
    """
    Class responsible for playing the site location game.

    Hackathon participants should create their AI class by inheriting from
    this class and overriding the place_stores method
    """

    def __init__(self, player_id: int, config: Dict):
        self.player_id = player_id
        self.config = config

        self.name = f"{self.__class__.__name__}-{self.player_id}"
        self.color = self._get_color()
        self.stores_to_place: List[Store] = []

    def place_stores(
        self,
        slmap: SiteLocationMap,
        store_locations: Dict[int, List[Store]],
        current_funds: float,
    ):
        """Sets self.stores_to_place to a list of store locations to place
        into the slmap.  Stores at the beginning of the list have priority if
        there are not enough funds to place all of them.

        Arguments:
        - slmap: SiteLocationMap for the current round
        - store_locations: currently exisiting stores for all players, by id
          e.g. your current stores are: store_locations[self.player_id]
        - current_funds: amount of money available to spend on stores

        Note the game configuration will be available through self.config

        See ./example_players.py for basic example implementations.
        """
        raise NotImplementedError()

    def _get_color(self) -> Tuple[int, int, int]:
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        return colors[self.player_id % len(colors)]


DEFAULT_CONFIGURATION = {
    "map_size": (400, 400),
    "population": 1e6,
    "n_rounds": 10,
    "starting_cash": 70000,
    "profit_per_customer": 0.5,
    "max_stores_per_round": 2,
    "place_stores_time_s": 10,
    "ignore_player_exceptions": True,
    "store_config": {
        "small": {
            "capital_cost": 10000.0,
            "operating_cost": 1000.0,
            "attractiveness": 25.0,
            "attractiveness_constant": 1.0,
        },
        "medium": {
            "capital_cost": 50000.0,
            "operating_cost": 2000.0,
            "attractiveness": 50.0,
            "attractiveness_constant": 1.0,
        },
        "large": {
            "capital_cost": 100000.0,
            "operating_cost": 3000.0,
            "attractiveness": 100.0,
            "attractiveness_constant": 1.0,
        },
    },
}


def manhatten_distances(size, point):
    """Returns a numpy array of size size, with manhatten distances from the
    given point for every location in the array
    """
    x = np.linspace(0, size[0], size[0])
    y = np.linspace(0, size[1], size[1])

    distances = abs(x[:, None] - point[0]) + abs(y[None, :] - point[1])
    return distances


def euclidian_distances(size, point):
    """Returns a numpy array of size size, with euclidian distances from the
    given point for every location in the array
    """
    x = np.linspace(0, size[0], size[0])
    y = np.linspace(0, size[1], size[1])

    distances = np.sqrt(
        np.square(x[:, None] - point[0]) + np.square(y[None, :] - point[1])
    )
    return distances


def closest_store_allocation(
    slmap: SiteLocationMap,
    players: Dict[int, SiteLocationPlayer],
    stores: Dict[int, List[Store]],
    store_config=None,
    max_dist=50,
):
    # TODO handle case when no stores created
    distances_by_player = {}
    global_min = None
    for player_id, player in players.items():
        least_distance = None
        for store in stores[player_id]:
            distances = manhatten_distances(slmap.size, store.pos)
            if least_distance is None:
                least_distance = distances
            else:
                least_distance = np.minimum(least_distance, distances)
        distances_by_player[player_id] = least_distance
        if global_min is None:
            global_min = least_distance
        else:
            global_min = np.minimum(least_distance, global_min)

    player_allocations = {}
    for player_id in players:
        player_least_distances = distances_by_player[player_id]
        player_allocations[player_id] = (
            (player_least_distances <= global_min)
            & (player_least_distances <= max_dist)
        ).astype(float)

    return player_allocations


def attractiveness_allocation(
    slmap: SiteLocationMap,
    stores: Dict[int, List[Store]],
    store_config: Dict[str, Dict[str, float]],
) -> Dict[int, np.ndarray]:
    """Returns population allocation for the given map, players and stores.

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
    total_attractiveness = np.zeros(slmap.size)
    for player_id in stores:
        best_attractiveness = np.zeros(slmap.size)
        for store in stores[player_id]:
            distances = euclidian_distances(slmap.size, store.pos)
            attractiveness = (
                store_config[store.store_type]["attractiveness"]
                / np.maximum(distances, np.ones(distances.shape))
                - store_config[store.store_type]["attractiveness_constant"]
            )
            attractiveness = np.where(attractiveness < 0, 0, attractiveness)
            best_attractiveness = np.maximum(best_attractiveness, attractiveness)
        attractiveness_by_player[player_id] = best_attractiveness
        total_attractiveness += best_attractiveness
    total_attractiveness = np.where(total_attractiveness == 0, 1, total_attractiveness)

    player_allocations = {}
    for player_id in stores:
        allocation = attractiveness_by_player[player_id] / total_attractiveness
        player_allocations[player_id] = allocation

    return player_allocations


class PlayerTimedOutError(RuntimeError):
    pass


def timeout_handler(signum, frame):
    raise PlayerTimedOutError()


class SiteLocationGame:
    """
    Class controlling the site location game.
    """

    def __init__(self, config: Dict, player_classes: List[type], allocation_func):
        self.allocation_func = allocation_func
        self.config = config
        self.timeouts = 0
        self.store_type_error = False
        self.out_of_bounds_error = False

        # Note - all of the below attributes follow the same pattern of being
        # lists, where each entry represents the state during a given round
        # i.e. self.store_locations[3] will return the stores for each player
        # as they were on the 3rd round
        log.info("Initializing Map")
        self.slmaps = [
            SiteLocationMap(config["map_size"], population=config["population"])
        ]

        log.info("Initializing Players")
        self.players: Dict[int, SiteLocationPlayer] = {}
        self.store_locations: List[Dict[int, List[Store]]] = [{}]
        self.allocations: List[Dict[int, np.ndarray]] = [{}]
        self.scores: List[Dict[int, float]] = [{}]

        for i, player_class in enumerate(player_classes):
            try:
                self.players[i] = player_class(i, config)
            except Exception as e:
                log.error(f"Failed to instantiate player {i}")
            self.store_locations[0][i] = []
            self.allocations[0][i] = np.zeros(config["map_size"])
            self.scores[0][i] = config["starting_cash"]

        self.current_round = 0

    def play(self):
        """Plays a full site location game, returns the winning
        SiteLocationPlayer object.
        """
        log.info("Starting game")
        for i in range(self.config["n_rounds"]):
            self.play_round()
        log.info(f"Winner: {self.winner().name}")
        return self.winner()

    def play_round(self):
        """Plays a single round of the site location game"""
        self.current_round += 1
        log.info(f"Starting round {self.current_round}")

        self.slmaps.append(deepcopy(self.slmaps[-1]))

        self.store_locations.append({})
        store_costs = {}
        for player_id, player in self.players.items():
            prev_score = self.scores[-1][player_id]
            player.stores_to_place = []
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.config["place_stores_time_s"])
            except AttributeError:
                # We're on windows, so we can't use SIGALRM to limit execution
                # time
                pass
            start_time = time.time()
            if self.config["ignore_player_exceptions"]:
                try:
                    player.place_stores(
                        deepcopy(self.slmaps[-1]), self.store_locations[-2], prev_score
                    )
                    signal.alarm(0)  # clear current alarm
                except PlayerTimedOutError:
                    log.warn(f"Player {player.name} timed out placing stores")
                    self.timeouts += 1
                    new_stores = player.stores_to_place
                except Exception as e:
                    log.warn(f"Player {player.name} raised exception in place_stores")
                    new_stores = []
            else:
                player.place_stores(
                    deepcopy(self.slmaps[-1]), self.store_locations[-2], prev_score
                )
                signal.alarm(0)  # clear current alarm

            elapsed = time.time() - start_time
            if elapsed > self.config["place_stores_time_s"]:
                log.warn(
                    f"Player {player.name} exceeded time limit placing stores {elapsed:.2f}"
                )
            else:
                log.debug(f"Player {player.name} took {elapsed:.2f}s to place stores")

            new_stores = player.stores_to_place

            valid_stores = self.valid_stores(new_stores, prev_score)
            for store in valid_stores:
                log.debug(
                    f"Player {player.name} placed a {store.store_type} store at {store.pos}"
                )
            if new_stores != valid_stores:
                log.debug(
                    f"Player {player.name} attempted to place {len(new_stores)} store(s), but was only able to place {len(valid_stores)}"
                )
            else:
                log.debug(f"Player {player.name} placed {len(new_stores)} store(s)")
            new_stores = valid_stores

            all_stores = self.store_locations[-2][player_id] + new_stores
            self.store_locations[-1][player_id] = all_stores
            store_costs[player_id] = self.store_cost(new_stores, all_stores)

        allocations = self.allocation_func(
            self.slmaps[-1], self.store_locations[-1], self.config["store_config"]
        )
        self.allocations.append(allocations)

        round_score = self.round_score()
        self.scores.append({})
        for player_id, player in self.players.items():
            prev_score = self.scores[-2][player_id]
            new_score = round_score[player_id]
            cost = store_costs[player_id]
            log.info(
                f"Player {player.name} earned ${new_score:.2f} and spent ${cost:.2f}"
            )
            current_score = prev_score + new_score - cost
            self.scores[-1][player_id] = current_score
            log.info(f"Player {player.name} has ${current_score:.2f}")

    def valid_stores(self, new_stores, current_score):
        """Returns the list of stores in new_stores that can be afforded with
        current_score. Also limits the max number of stores by the self.config
        """
        valid_stores = []
        new_stores = new_stores.copy()
        max_stores = self.config["max_stores_per_round"]
        while (
            current_score >= 0 and len(valid_stores) <= max_stores and len(new_stores)
        ):
            store = new_stores.pop(0)
            if store.store_type not in list(self.config["store_config"].keys()):
                self.store_type_error = True
                msg = f"Player attempted to place invalid store type"
                log.warn(msg)
                raise RuntimeError(msg)
            elif (
                store.pos[0]
                > self.config["map_size"][0] | store.pos[1]
                > self.config["map_size"][1]
            ):
                self.out_of_bounds_error = True
                msg = f"Player attempted to place store out of bounds"
                log.warn(msg)
                raise RuntimeError(msg)
            cost = self.config["store_config"][store.store_type]["capital_cost"]
            if cost <= current_score:
                valid_stores.append(store)
                current_score -= cost
        return valid_stores

    def store_cost(self, new_stores, all_stores):
        """
        Calculate cost of building new_stores and operating all_stores
        """
        cost = 0.0
        for store in new_stores:
            cost += self.config["store_config"][store.store_type]["capital_cost"]

        for store in all_stores:
            cost += self.config["store_config"][store.store_type]["operating_cost"]
        return cost

    def round_score(self, round_number=-1):
        """Return the amount of revenue earned by each player in the given round"""
        scores = {}
        for player_id in self.players:
            new_score = (
                np.sum(
                    self.slmaps[round_number].population_distribution
                    * self.allocations[round_number][player_id]
                )
                * self.config["profit_per_customer"]
            )
            scores[player_id] = new_score
        return scores

    def winner(self) -> SiteLocationPlayer:
        """Return the player object with the highest score in the most recent
        round.
        """
        player_id = max(self.scores[-1], key=self.scores[-1].get)
        return self.players[player_id]

    def get_scores(self) -> List[Tuple[SiteLocationPlayer, float]]:
        """Returns a list of (player, score) tuples

        Score in this case is the % of final money between all players
        """

        total_cash = sum(self.scores[-1].values())
        if total_cash <= 0:
            return [(p, 0.0) for p in self.players.values()]
        else:
            return [
                (self.players[player_id], self.scores[-1][player_id] / total_cash)
                for player_id in self.players
            ]

    def save_image(self, filename, round_number=-1):
        """Save an image of the currrent game round to filename, for the given
        round (by default the latest round)
        """
        slmap = self.slmaps[round_number]
        stores = self.store_locations[round_number]
        allocations = self.allocations[round_number]

        slmap.save_image(
            filename, players=self.players, stores=stores, allocations=allocations
        )

    def save_game_report(self, dirname):
        """Create a game report directory with the following contents:
        - images of each round
        - plots of score/time
        - markdown report with game details/configuration
        """
        log.info(f"Saving game report to: {dirname}")
        try:
            shutil.rmtree(dirname)
        except FileNotFoundError:
            pass

        os.makedirs(dirname)

        # save each round image
        for round_number in range(self.current_round + 1):
            round_image_filename = os.path.join(
                dirname, f"map-round-{round_number:02}.png"
            )
            self.save_image(round_image_filename, round_number)

        # Plot scores over time
        fig, ax = plt.subplots()
        rounds = list(range(self.current_round + 1))
        for player_id, player in self.players.items():
            ax.plot(
                rounds,
                [self.scores[i][player_id] for i in rounds],
                label=player.name,
                color=[c / 255.0 for c in player.color],
            )
        ax.set_xlabel("Round #")
        ax.set_ylabel("Funds ($)")
        ax.set_title("Funds over all Rounds")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(dirname, "scores.png"))

        # Create markdown report
        with open(os.path.join(dirname, "game_report.md"), "w") as f:
            f.write("# Game Report\n")
            f.write("## Game configuration\n")
            f.write(f"allocation_function: {self.allocation_func.__name__}\n\n")
            for option, value in self.config.items():
                f.write(f"{option}: {value}\n\n")

            f.write("## Players\n")
            for player_id, player in self.players.items():
                f.write(f"- {player.name}\n")

            final_scores = self.scores[-1]
            f.write("## Final Scores\n")
            for player_id, player in self.players.items():
                score = final_scores[player_id]
                f.write(f"{player.name}: {score}\n\n")

            f.write("# Winner!\n")
            f.write(f"{self.winner().name}\n")


def import_player(player_str):
    """Return the requested class

    player_str - string formatted like <module>:<class>
    """
    module = player_str.split(":")[0]
    classname = player_str.split(":")[1]

    mod = __import__(module, fromlist=classname)
    return getattr(mod, classname)


def main():

    parser = argparse.ArgumentParser(description="Site Location Game")
    parser.add_argument(
        "--players",
        nargs="+",
        type=str,
        help="pass a series of <module>:<class> strings to specify the players in the game",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="game",
        help="report game results to the given dir",
    )
    args = parser.parse_args()

    if args.players is None:
        parser.print_help()
        exit(-1)

    players = []
    for player_str in args.players:
        players.append(import_player(player_str))

    game = SiteLocationGame(DEFAULT_CONFIGURATION, players, attractiveness_allocation)
    game.play()
    game.save_game_report(args.report)


if __name__ == "__main__":
    main()
