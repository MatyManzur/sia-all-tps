from __future__ import annotations
from typing import List
from typing import Callable
import copy

INVALID = 0
OK = 1

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
    def __init__(self, sokoban_map: List[List[0 | 1]], goals: List[Position]):
        self.map = sokoban_map
        self.goals = goals
        self.blocked_positions_map = copy.deepcopy(sokoban_map)

        for y in range(len(sokoban_map)):
            for x in range(len(sokoban_map[0])):
                if sokoban_map[y][x] == 0:
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
        new_state = copy.deepcopy(self)
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
            print('Intentando mover contra el borde')
            return None

        if board.map[new_state.player_position.y][new_state.player_position.x] == 1:  # si es una pared
            print('Intentando mover contra la pared')
            return None

        return new_state

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
                print('Estoy intentando mover una caja contra otra caja')
                return None

            # Si la caja quedó en un estado bloqueado
            if self.board.blocked_positions_map[new_box_position.y][new_box_position.x] == BLOCKED_BOTH:
                print('La caja está en un estado bloquedo')
                return None

            new_state.box_positions.remove(new_state.player_position)
            new_state.box_positions.append(new_box_position)

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

    def get_children(self, board: Board) -> List[Node]:
        new_nodes = []
        directions = ['up', 'down', 'right', 'left']
        for direc in directions:
            state = self.state.try_move(board, direc)
            if state is not None:
                new_node = Node(state, self.cost + 1, self)
                new_nodes.append(new_node)
        return new_nodes

    def get_path_from_root(self) -> List[State]:
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


class SolutionInfo:
    def __init__(self, path_to_solution: List[State], final_cost: int, expanded_nodes_count: int, frontier_nodes_count: int):
        self.path_to_solution = path_to_solution
        self.final_cost = final_cost
        self.expanded_nodes_count = expanded_nodes_count
        self.frontier_nodes_count = frontier_nodes_count


class Algorithm:
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
        return self

    def __next__(self):
        if self.has_finished():
            return self.solution

        node = None
        while self.frontier:
            node = self.frontier.pop()  # va a sacar el último
            saved_score = self.visited.get(node, float('inf'))  # sería como visited.getOrDefault(node,Math.Inf) de Java
            if not self._visited_value(node) >= saved_score:
                break

        if node is None:
            self.no_solution = True
            return None

        if node.state.is_goal_state():
            print("GANASTEEEE WIIIIIII!")
            self.no_solution = False
            self.solution = node
            return self.solution

        # si no es un goal, lo agregamos a visited
        self.visited[node] = self._visited_value(node)
        # y lo expandimos
        children = node.get_children(self.board)
        for child in children:
            if not child.state.is_blocked:
                self._add_to_frontier(child)
        return node

    def has_finished(self) -> bool:
        return self.solution is not None

    def has_solution(self) -> bool:
        return not self.no_solution

    def get_solution_info(self) -> SolutionInfo:
        return SolutionInfo(1,1,1,1)

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


