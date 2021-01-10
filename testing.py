import argparse
import logging
import random

import numpy as np
from multiprocessing import Pool

logging.basicConfig()
log = logging.getLogger("site_location")
log.setLevel(logging.ERROR)

from site_location import SiteLocationGame, import_player, attractiveness_allocation


DEFAULT_CONFIGURATION = {
    "map_size": (400, 400),
    "population": 1e6,
    "n_rounds": 10,
    "starting_cash": 70000,
    "profit_per_customer": 0.5,
    "max_stores_per_round": 2,
    "place_stores_time_s": -1,
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


def play_game(ii):
    seed = seeds[ii]
    random.seed(seed)
    np.random.seed(seed)
    game = SiteLocationGame(DEFAULT_CONFIGURATION, players, attractiveness_allocation)
    game.play()
    game.save_game_report(f"game/{ii}")
    return game.get_scores(), game.round_score()


def process_results(res):
    percent_scores = [t[0] for t in res]
    money_scores = [t[1] for t in res]

    money_total = money_scores.pop(0)

    for d in money_scores:
        for k in d.keys():
            money_total[k] += d[k]

    print(money_total)

    average_percent = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for t in percent_scores:
        for i in range(len(t)):
            average_percent[i] += t[i][1] / n

    print(average_percent)


if __name__ == "__main__":

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

    n = 10
    seed = 69420
    random.seed(seed)

    seeds = [random.randint(0, seed) for _ in range(n)]

    player_scores = []
    player_percentages = []

    p = Pool(10)
    res = p.map(play_game, list(range(0, n)))
    process_results(res)
