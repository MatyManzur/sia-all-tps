from __future__ import annotations
from typing import List
from typing import Callable
from algorithm_utils import Position, Board


def trivial_heuristic(player_position: Position, box_positions: List[Position], board: Board) -> int:
    for box_pos in box_positions:
        if box_pos not in board.goals:
            return 1
    return 0


def manhattan_heuristic(player_position: Position, box_positions: List[Position], board: Board) -> int:
    min_distance = {}
    for goals in board.goals:
        for box_pos in box_positions:
            if box_pos not in board.goals:
                distance = abs(goals.x - box_pos.x) + abs(goals.y - box_pos.y)
                if distance < min_distance.get(box_pos, float('inf')):
                    min_distance[box_pos] = distance
    distance_sum = 0
    for value in min_distance.values():
        distance_sum += value
    return distance_sum
