from __future__ import annotations

import copy
from typing import List, Set, Dict, Callable
from algorithm_utils import Position, Board


def trivial_heuristic(player_position: Position, box_positions: Set[Position], board: Board) -> int:
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


def metro_heuristic(player_position: Position, box_positions: Set[Position], board: Board) -> int:
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


class PreCalcHeuristic:
    def __init__(self):
        self.initialized = False
        self.goal_cost_maps: Dict[Position, Dict[Position, int]] = {}

    @staticmethod
    def __is_open_space(wall_map: List[List[0 | 1]], x: int, y: int):
        return 0 <= y < len(wall_map) and 0 <= x < len(wall_map[0]) and wall_map[y][x] == 0

    def __get_possible_previous_positions(self, wall_map: List[List[0 | 1]], pos: Position) -> List[Position]:
        possible_prev_pos = []
        if self.__is_open_space(wall_map, pos.x, pos.y + 1) and self.__is_open_space(wall_map, pos.x, pos.y + 2):
            possible_prev_pos.append(Position(pos.x, pos.y + 1))
        if self.__is_open_space(wall_map, pos.x, pos.y - 1) and self.__is_open_space(wall_map, pos.x, pos.y - 2):
            possible_prev_pos.append(Position(pos.x, pos.y - 1))
        if self.__is_open_space(wall_map, pos.x + 1, pos.y) and self.__is_open_space(wall_map, pos.x + 2, pos.y):
            possible_prev_pos.append(Position(pos.x + 1, pos.y))
        if self.__is_open_space(wall_map, pos.x - 1, pos.y) and self.__is_open_space(wall_map, pos.x - 2, pos.y):
            possible_prev_pos.append(Position(pos.x - 1, pos.y))
        return possible_prev_pos

    def __estimate_cost_for_goal(self, wall_map: List[List[0 | 1]], cost_map: Dict[Position, int], position: Position,
                                 potential_position_cost: int):
        possible_prev_pos = self.__get_possible_previous_positions(wall_map, position)
        cost_map[position] = potential_position_cost
        for prev_pos in possible_prev_pos:
            if cost_map.get(prev_pos, float('inf')) > potential_position_cost + 1:
                self.__estimate_cost_for_goal(wall_map, cost_map, prev_pos, potential_position_cost + 1)

    def __initialize(self, board: Board):
        for goal in board.goals:
            self.goal_cost_maps[goal] = {}
            self.__estimate_cost_for_goal(board.map, self.goal_cost_maps[goal], goal, 0)
            """# PRINT COST MAP FOR EACH GOAL
            for y in range(len(board.map) - 1, -1, -1):
                print("[ ", end="")
                for x in range(len(board.map[0])):
                    print(f"{self.goal_cost_maps[goal].get(Position(x, y), 'X')}, ", end="")
                print("]")
            print("-------------")
            """
        self.initialized = True

    def pre_calc_heuristic(self, player_position: Position, box_positions: Set[Position], board: Board) -> int:
        if not self.initialized:
            self.__initialize(board)

        min_distance = float('inf')
        for box in box_positions:
            distance = manhattan_distance(box, player_position)
            if distance < min_distance:
                min_distance = distance
        total_distance = min_distance

        for box in box_positions:
            min_distance = float('inf')
            for goal in board.goals:
                distance = self.goal_cost_maps[goal].get(box, float('inf'))
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance

        return total_distance


HEURISTICS = {
    'trivial': trivial_heuristic,
    'manhattan': manhattan_heuristic,
    'metro': metro_heuristic,
    'pre_calc': PreCalcHeuristic().pre_calc_heuristic
}


def get_heuristic(name: str) -> Callable[[Position, Set[Position], Board], int]:
    return HEURISTICS[name]
