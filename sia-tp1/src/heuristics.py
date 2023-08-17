from __future__ import annotations
from typing import List
from typing import Callable
from algorithm_utils import Position, Board


def trivial_heuristic(player_position: Position, box_positions: List[Position], board: Board) -> int:
    for box_pos in box_positions:
        if box_pos not in board.goals:
            return 1
    return 0
