from __future__ import annotations
from typing import List, Tuple, Set
from typing import Callable
import copy
import numpy as np

INVALID = 0
OK = 1

BOX = 2
GOAL = 3
PLAYER = 4

BLOCKED_VERT = 2
BLOCKED_HOR = 3
BLOCKED_BOTH = 5


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
    def __init__(self, sokoban_map: List[List[0 | 1]], goals: set[Position]):
        self.map = np.array(sokoban_map)
        self.goals = goals
        self.blocked_positions_map = np.zeros((len(sokoban_map), len(sokoban_map[0])), dtype=int)

        for y in range(len(sokoban_map)):
            for x in range(len(sokoban_map[0])):
                if sokoban_map[y][x] == 0 and Position(x, y) not in goals:
                    position = Position(x, y)
                    blocked_vertical = (not (0 < position.y < len(self.map) - 1)) or (
                            self.map[position.y + 1][position.x] +
                            self.map[position.y - 1][position.x] != 0)
                    blocked_horizontal = (not (0 < position.x < len(self.map[0]) - 1)) or (
                            self.map[position.y][position.x - 1] +
                            self.map[position.y][position.x + 1] != 0)
                    if blocked_vertical:
                        self.blocked_positions_map[y][x] += BLOCKED_VERT
                    if blocked_horizontal:
                        self.blocked_positions_map[y][x] += BLOCKED_HOR


def get_positions(map_: List[List[int]]) -> Tuple[Board, Position, set[Position]]:
    # 0 camino, 1 pared, 2 caja, 3 goal, 4 player
    player = Position(0, 0)
    goals = set([])
    boxes = set([])
    for i in range(len(map_)):
        for j in range(len(map_[i])):
            cell = map_[i][j]
            if cell == PLAYER:
                player = Position(j, i)
            elif cell == BOX:
                boxes.add(Position(j, i))
            elif cell == GOAL:
                goals.add(Position(j, i))

            if cell != 1:
                map_[i][j] = 0
    return Board(map_, goals), player, boxes


class State:
    # Player position: position
    # Box positions: [position1, position2, ...]
    # Heuristic: function(player_position, box_positions, board)
    def __init__(self, player_position: Position, box_positions: set[Position],
                 heuristic: Callable[[Position, set[Position], Board], int], board: Board):
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
        new_x = self.player_position.x
        new_y = self.player_position.y
        if direction == 'up':
            new_y += 1
        elif direction == 'down':
            new_y -= 1
        elif direction == 'left':
            new_x -= 1
        elif direction == 'right':
            new_x += 1

        if (not (0 <= new_y < len(board.map)) or
                not (0 <= new_x < len(board.map[0]))):
            return None

        if board.map[new_y][new_x] == 1:  # si es una pared
            return None
        new_box_positions = set()
        for box_pos in self.box_positions:
            new_box_positions.add(Position(box_pos.x, box_pos.y))
        return State(Position(new_x, new_y), new_box_positions, self.heuristic_function, board)

    def try_move(self, board: Board, direction: str) -> State | None:
        new_state = self.__check_move(board, direction)
        # recalculamos la heurística porque hicimos una copia del estado

        if new_state is None:
            return None

        new_state.heuristic_value = new_state.heuristic_function(new_state.player_position, new_state.box_positions,
                                                                 new_state.board)
        if new_state.player_position in self.box_positions:
            aux_state = new_state.__check_move(board, direction)
            if aux_state is None:  # estoy intentando mover una caja contra una pared
                return None

            new_box_position = aux_state.player_position  # This is fine
            if new_box_position in self.box_positions:  # estoy empujando una caja contra otra caja
                return None

            # Si la caja quedó en un estado bloqueado
            if self.board.blocked_positions_map[new_box_position.y][new_box_position.x] == BLOCKED_BOTH:
                return None

            new_state.box_positions.remove(new_state.player_position)
            new_state.box_positions.add(new_box_position)

        return new_state

    def is_goal_state(self):
        for box_pos in self.box_positions:
            if box_pos not in self.board.goals:
                return False
        return True

    def __str__(self):
        return f"Player: {self.player_position}, Boxes: {self.box_positions}, Heuristic: {self.heuristic_value}"


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

    def get_children(self, board: Board) -> set[Node]:
        new_nodes = set([])
        directions = ['up', 'down', 'right', 'left']
        for direc in directions:
            state = self.state.try_move(board, direc)
            if state is not None:
                new_node = Node(state, self.cost + 1, self)
                new_nodes.add(new_node)
        return new_nodes

    def get_path_from_root(self) -> set[State]:
        path = []
        node = self
        while node is not None:
            path.insert(0, node.state)
            node = node.parent
        return path

    def __hash__(self) -> int:
        return hash(self.state)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Node):
            return self.state == __o.state
        else:
            return False

    def __lt__(self, other: Node) -> bool:
        return self.score < other.score


class SolutionInfo:
    def __init__(self, path_to_solution: Set[State], final_cost: int, expanded_nodes_count: int,
                 frontier_nodes_count: int):
        self.path_to_solution = path_to_solution
        self.final_cost = final_cost
        self.expanded_nodes_count = expanded_nodes_count
        self.frontier_nodes_count = frontier_nodes_count


class Algorithm:
    def __init__(self, board: Board, player_position: Position, box_positions: set[Position],
                 heuristic: Callable[[Position, set[Position], Board], int], sort_children: bool = False,
                 sort_children_key: Callable[[Node], int] = None):
        self.frontier: set[Node] = []
        self.visited = {}
        self.initial_state = State(player_position, box_positions, heuristic, board)
        self.board = board
        self.no_solution = None
        self.solution: Node | None = None
        self.sort_frontier = sort_children
        self.sort_children_key = sort_children_key

    def __iter__(self):
        self.frontier.append(Node(self.initial_state, 0, None))
        return self

    def __next__(self):
        if self.has_finished():
            return self.solution

        node = None
        while self.frontier:
            node = self._get_item_from_frontier()  # va a sacar el último
            saved_score = self.visited.get(node, float('inf'))  # sería como visited.getOrDefault(node,Math.Inf) de Java
            if not self._visited_value(node) >= saved_score:
                break

        if node is None:
            self.no_solution = True
            return None

        if node.state.is_goal_state():
            print("GANASTEEEE 🎉🎉🎇✨🎊🎊 !")
            self.no_solution = False
            self.solution = node
            return self.solution

        # si no es un goal, lo agregamos a visited
        self.visited[node] = self._visited_value(node)
        # y lo expandimos
        children = node.get_children(self.board)
        if self.sort_frontier:
            children = sorted(children, key=self.sort_children_key, reverse=True)
        for child in children:
            if not child.state.is_blocked:
                self._add_to_frontier(child)
        return node

    def has_finished(self) -> bool:
        return self.solution is not None

    def has_solution(self) -> bool:
        return not self.no_solution

    def get_solution_info(self) -> SolutionInfo:
        if not self.has_solution():
            raise 'A solution must be found to return info!'
        return SolutionInfo(self.solution.get_path_from_root(), self.solution.cost, len(self.visited.keys()),
                            len(self.frontier))

    # con esto determinamos si se comporta como una cola o lista para BFS y DFS
    # o como una lista ordenada por algún criterio como A*
    # (el próximo nodo a usar tiene que quedar al final)
    def _add_to_frontier(self, new_node: Node):
        """abstract method"""

    # para el caso de A*, visited_value() sería una función que
    # devuelva el A* score, para los otros sería una lambda que siempre devuelve 1
    # no sé si está bien lo de @staticmethod, pero sería un static en java,
    # pero que a veces lo puedas overridear en algún hijo
    @staticmethod
    def _visited_value(node: Node) -> int:
        return 1

    def _get_item_from_frontier(self) -> Node:
        return self.frontier.pop()
