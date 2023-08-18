from __future__ import annotations

import copy
from typing import List,Set
from typing import Callable
from algorithm_utils import Position, Board


def trivial_heuristic(player_position: Position, box_positions: List[Position], board: Board) -> int:
    for box_pos in box_positions:
        if box_pos not in board.goals:
            return 1
    return 0


def manhattan_distance(pos_1: Position, pos_2: Position):
    return abs(pos_1.x - pos_2.x) + abs(pos_1.y - pos_2.y)


def manhattan_heuristic(player_position: Position, box_positions: Set[Position], board: Board) -> int:
    min_distance = {}
    for goal in board.goals:
        for box_pos in box_positions:
            if box_pos not in board.goals:
                distance = manhattan_distance(goal, box_pos)
                if distance < min_distance.get(box_pos, float('inf')):
                    min_distance[box_pos] = distance
    distance_sum = 0
    for value in min_distance.values():
        distance_sum += value
    return distance_sum

def better_manhattan_heuristic(player_position: Position, box_positions: List[Position], board: Board) -> int:
    current_position = player_position
    total_distance = 0
    boxes = copy.copy(box_positions)
    goals = copy.copy(board.goals)
    while len(boxes) > 0 and len(goals) > 0:
        minimum_distance = float('inf')
        min_box = None
        for box in boxes:
            dist = manhattan_distance(current_position, box)
            if dist < minimum_distance:
                minimum_distance = dist
                min_box = box
        current_position = min_box
        boxes.remove(min_box)
        total_distance += minimum_distance
        minimum_distance = float('inf')
        min_goal = None
        for goal in goals:
            dist = manhattan_distance(current_position, goal)
            if dist < minimum_distance:
                minimum_distance = dist
                min_goal = goal
        current_position = min_goal
        total_distance += minimum_distance
        goals.remove(min_goal)
    return total_distance

