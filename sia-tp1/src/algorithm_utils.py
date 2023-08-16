from __future__ import annotations
from typing import List, Tuple
from abc import ABC, abstractmethod
from typing import Callable
import copy

INVALID = 0
OK = 1


class Position:
    # Position in 2D space
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Position):
            if self.x == __o.x and self.y == __o.y:
                return True
        return False


class Board:
    # map of the board
    # goals denoted as [position1, position2, ...]
    def __init__(self, sokoban_map, goals):
        self.map = sokoban_map
        self.goals = goals


class State:
    # Player position: position
    # Box positions: [position1, position2, ...]
    # Heuristic: function(player_position,Box_positions)
    def __init__(self, player_position: Position, box_positions: List[Position],
                 heuristic: Callable[[Position, List[Position]], int]):
        self.player_position = player_position
        self.box_positions = box_positions
        self.heuristic_value = heuristic(player_position, box_positions)

    def __hash__(self) -> int:
        return hash((self.player_position, tuple(self.box_positions)))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, State):
            if self.player_position.__eq__(__o.player_position):
                for box in self.box_positions:
                    if box not in __o.box_positions:
                        return False
                return True
        else:
            return False

    def __check_move(self, board: Board, direction: str) -> State | None:
        new_state = copy.copy(self)
        if direction == 'up':
            new_state.player_position.y += 1
        elif direction == 'down':
            new_state.player_position.y -= 1
        elif direction == 'left':
            new_state.player_position.x -= 1
        elif direction == 'right':
            new_state.player_position.x += 1

        if (not (0 <= new_state.player_position.y < len(board.map)) or
                not (0 <= new_state.player_position.x < len(board.map[0]))):
            return None

        if board.map[new_state.player_position.y][new_state.player_position.x] == 1:  # si es una pared
            return None

        return new_state

    def try_move(self, board: Board, direction: str) -> State | None:
        new_state = self.__check_move(board, direction)
        if new_state is None:
            return None

        if new_state.player_position in self.box_positions:
            aux_state = new_state.__check_move(board, direction)
            if aux_state is None:  # estoy intentando mover una caja contra una pared
                return None

            new_box_position = aux_state.player_position
            if new_box_position in self.box_positions:  # estoy empujando una caja contra otra caja
                return None
                
            new_state.box_positions.remove(new_state.player_position)
            new_state.box_positions.append(new_box_position)

        # Check if there is a box on the new position and try to move it
        # if we can move it, we have to change the state of the boxes in our new state
        # for box_position in self.box_positions:
        #     if position == box_position:
        #
        #
        #
        # if position in self.box_positions:
        #     (result, new_box_position) = self.try_move(board, new_position, direction)
        #     if result == INVALID:
        #         return INVALID, self


class Node:
    # State is the program state
    # Cost is the cost of the path plus the current cost
    # Score is the cost plus the heuristic
    # Parent is the parent node in the tree
    def __init__(self, state: State, cost: int, score: int, parent: Node | None):
        self.state = state
        self.cost = cost
        self.score = score
        self.parent = parent

    def get_children(self, board: Board) -> List[Node]:
        frontier = []
        directions = ['up', 'down', 'right', 'left']  # habría que ponerlo afuera
        p = self.state.player_position
        for dir in directions:
            state = self.state.try_move(board, dir)
            if state is not None:
                newnode = Node(state, 1, 1, self)  # ver lo de cost y score
                frontier.append(newnode)
        return frontier


class Algorithm(ABC):
    def __init__(self, initial_state: State, board: Board, heuristic: Callable[[State], int],
                 cost_func: Callable[[State], int]):
        self.heuristic = heuristic
        self.cost_func = cost_func
        self.frontier = []
        self.initial_state = initial_state
        self.board = board

    def __iter__(self):
        self.frontier = [
            Node(self.initial_state, self.cost_func(self.initial_state), self.heuristic(self.initial_state), None)]

    @abstractmethod
    def __next__(self):
        """To implement"""

    @abstractmethod
    def has_finished(self) -> bool:
        """To implement"""

    @abstractmethod
    def has_solution(self) -> bool:
        """To implement"""


class BFSAlgorithm(Algorithm):
    def __init__(self, initial_state: State, board: Board, heuristic: Callable[[State], int],
                 cost_func: Callable[[State], int]):
        super().__init__(initial_state, board, heuristic, cost_func)

    def __next__(self):
        """ PUSE A IMPLEMENTAR BFS DEBERIA SER BASTANTE FACIL """
