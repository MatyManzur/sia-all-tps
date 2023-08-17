from __future__ import annotations
from typing import List
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
    # Heuristic: function(player_position, box_positions, board)
    def __init__(self, player_position: Position, box_positions: List[Position],
                 heuristic: Callable[[Position, List[Position], Board], int], board: Board):
        self.player_position = player_position
        self.box_positions = box_positions
        self.heuristic_function = heuristic
        self.heuristic_value = heuristic(player_position, box_positions, board)
        self.board = board
        self.is_blocked = False

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
        # recalculamos la heurística porque hicimos una copia del estado
        new_state.heuristic_value = new_state.heuristic_function(new_state.player_position, new_state.box_positions,
                                                                 new_state.board)
        if new_state is None:
            return None

        if new_state.player_position in self.box_positions:
            aux_state = new_state.__check_move(board, direction)
            if aux_state is None:  # estoy intentando mover una caja contra una pared
                return None

            new_box_position = aux_state.player_position  # This is fine
            if new_box_position in self.box_positions:  # estoy empujando una caja contra otra caja
                return None

            new_state.box_positions.remove(new_state.player_position)
            new_state.box_positions.append(new_box_position)
            new_state.__is_blocked(new_box_position)

        return new_state

    def is_goal_state(self):
        for box_pos in self.box_positions:
            if box_pos not in self.board.goals:
                return False
        return True

    # We could make this more efficient by calculating it only in case a box was moved
    def __is_blocked(self, new_box_position: Position):
        blocked_vertical = (not (0 < new_box_position.y < len(self.board.map) - 1)) or (
                    self.board.map[new_box_position.y + 1][new_box_position.x] + self.board.map[new_box_position.y - 1][
                new_box_position.x]) != 0
        blocked_horizontal = (not (0 < new_box_position.x < len(self.board.map[0]) - 1)) or (
                    self.board.map[new_box_position.y][new_box_position.x + 1] + self.board.map[new_box_position.y][
                new_box_position.x + 1]) != 0
        if not blocked_vertical:
            for box_pos in self.box_positions:
                if box_pos == new_box_position:
                    pass
                elif box_pos.y == new_box_position.y - 1 or box_pos.y == new_box_position.y + 1:
                    blocked_vertical = True
                    break
        if not blocked_horizontal:
            for box_pos in self.box_positions:
                if box_pos == new_box_position:
                    pass
                elif box_pos.x == new_box_position.x - 1 or box_pos.x == new_box_position.x + 1:
                    blocked_horizontal = True
                    break
        return blocked_vertical and blocked_horizontal


class Node:
    # State is the program state
    # Cost is the cost of the path plus the current cost
    # Score is the cost plus the heuristic
    # Parent is the parent node in the tree
    def __init__(self, state: State, cost: int, parent: Node | None):
        self.state = state
        self.cost = cost
        self.score = cost + state.heuristic_value
        self.parent = parent

    def get_children(self, board: Board) -> List[Node]:
        new_nodes = []
        directions = ['up', 'down', 'right', 'left']
        current_pos = self.state.player_position
        for direc in directions:
            state = self.state.try_move(board, direc)
            if state is not None:
                new_node = Node(state, self.cost + 1, self)
                new_nodes.append(new_node)
        return new_nodes

    def get_path_from_root(self) -> List[Node]:
        path = []
        node = self
        while node is not None:
            path.insert(0, node)
            node = node.parent
        return path


class Algorithm(ABC):
    def __init__(self, board: Board, player_position: Position, box_positions: List[Position],
                 heuristic: Callable[[Position, List[Position], Board], int]):
        self.frontier: List[Node] = []
        self.visited = {}
        self.initial_state = State(player_position, box_positions, heuristic, board)
        self.board = board
        self.no_solution = None
        self.solution = None

    def __iter__(self):
        self.frontier = [Node(self.initial_state, 0, None)]

    def __next__(self):
        if self.has_finished():
            return self.solution

        node = None
        while self.frontier:
            node = self.frontier.pop()  # va a sacar el último
            saved_score = self.visited.get(node, float('inf'))  # sería como visited.getOrDefault(node,Math.Inf) de Java
            if not self.__visited_value(node) >= saved_score:
                break

        if node is None:
            self.no_solution = True
            return None

        if node.state.is_goal_state():
            self.no_solution = False
            return self.solution

        # si no es un goal, lo agregamos a visited
        self.visited[node] = self.__visited_value(node)
        # y lo expandimos
        children = node.get_children(self.board)
        for child in children:
            if not child.state.is_blocked:
                self.__add_to_frontier(child)

    def has_finished(self) -> bool:
        return self.solution is not None

    def has_solution(self) -> bool:
        return not self.no_solution

    # con esto determinamos si se comporta como una cola o lista para BFS y DFS
    # o como una lista ordenada por algún criterio como A*
    # (el próximo nodo a usar tiene que quedar al final)
    @abstractmethod
    def __add_to_frontier(self, new_node: Node):
        """abstract method"""

    # para el caso de A*, visited_value() sería una función que
    # devuelva el A* score, para los otros sería una lambda que siempre devuelve 1
    # no sé si está bien lo de @staticmethod, pero sería un static en java,
    # pero que a veces lo puedas overridear en algún hijo
    @staticmethod
    def __visited_value(node: Node) -> int:
        return 1


class BFSAlgorithm(Algorithm):
    # no hay que pisar next(), solo add_to_frontier y visited_value (o dejar el default),
    # y init() si queremos que no use heuristica

    def __init__(self, board: Board, player_position: Position, box_positions: List[Position]):
        super().__init__(board, player_position, box_positions,
                         lambda _, __, ___: 0)  # Es desinformado => No usa heuristica

    def __add_to_frontier(self, new_node: Node):
        self.frontier.insert(0, new_node)


class DFSAlgorithm(Algorithm):

    def __init__(self, board: Board, player_position: Position, box_positions: List[Position]):
        super().__init__(board, player_position, box_positions,
                         lambda _, __, ___: 0)  # Es desinformado => No usa heuristica

    def __add_to_frontier(self, new_node: Node):
        self.frontier.append(new_node)


class AStarAlgorithm(Algorithm):

    def __init__(self, board: Board, player_position: Position, box_positions: List[Position],
                 heuristic: Callable[[Position, List[Position], Board], int]):
        super().__init__(board, player_position, box_positions, heuristic)

    @staticmethod
    def __visited_value(node: Node) -> int:
        return node.score

    def __add_to_frontier(self, new_node: Node):
        self.frontier.append(new_node)
        self.frontier.sort(key=lambda x: x.score,
                           reverse=True)  # reverse=True para que quede ordenado de mayor a menor
