import random
import operator

from site_location import (
    SiteLocationPlayer,
    Store,
    SiteLocationMap,
    euclidian_distances,
    attractiveness_allocation,
)


def rank_random_k_locations(k, heuristic_func, slmap):
    sample_pos = []
    for i in range(k):
        x = random.randint(0, slmap.size[0])
        y = random.randint(0, slmap.size[1])
        sample_pos.append((x, y))

    scores = [heuristic_func(pos) for pos in sample_pos]
    locations, scores = zip(*sorted(zip(sample_pos, scores)))

    return locations, scores


def build_game_tree(k, heuristic_func, slmap):
    locations, scores = rank_random_k_locations(k, heuristic_func, slmap)
