import queue
from algorithm_utils import *
from collections import deque

from algorithm_utils import Node


class BFSAlgorithm(Algorithm):
    # no hay que pisar next(), solo add_to_frontier y visited_value (o dejar el default),
    # y init() si queremos que no use heuristica
    def __init__(self, board: Board, player_position: Position, box_positions: Set[Position]):
        super().__init__(board, player_position, box_positions,
                         lambda _, __, ___: 0)  # Es desinformado => No usa heuristica
        self.frontier = deque()

    def _add_to_frontier(self, new_node: Node):
        self.frontier.appendleft(new_node)


class DFSAlgorithm(Algorithm):

    def __init__(self, board: Board, player_position: Position, box_positions: Set[Position]):
        super().__init__(board, player_position, box_positions,
                         lambda _, __, ___: 0)  # Es desinformado => No usa heuristica
        self.frontier = deque()

    def _add_to_frontier(self, new_node: Node):
        self.frontier.append(new_node)


class IDDFSAlgorithm(Algorithm):
    def __init__(self, board: Board, player_position: Position, box_positions: List[Position], depth_increment: int = 1):
        super().__init__(board, player_position, box_positions,
                         lambda _, __, ___: 0)  # Es desinformado => No usa heuristica
        self.frontier = deque()
        self.depth_increment = depth_increment
    
    def __iter__(self):
        super().__iter__()
        self.max_depth = 1
        self.start_node = self.frontier[0]
        return self


    def _add_to_frontier(self, new_node: Node):
        if (new_node.depth <= self.max_depth):
            self.frontier.append(new_node)
        else:
            if len(self.frontier) == 0:
                self.frontier.append(self.start_node)
                self.max_depth += self.depth_increment


class AStarAlgorithm(Algorithm):

    def __init__(self, board: Board, player_position: Position, box_positions: Set[Position],
                 heuristic: Callable[[Position, Set[Position], Board], int]):
        super().__init__(board, player_position, box_positions, heuristic)
        self.frontier = queue.PriorityQueue()

    @staticmethod
    def _visited_value(node: Node) -> int:
        return node.score

    def _add_to_frontier(self, new_node: Node):
        self.frontier.put((new_node.score, new_node))

    def _get_item_from_frontier(self) -> Node:
        return self.frontier.get()[1]

    def __iter__(self):
        aux = Node(self.initial_state, 0, None)
        self.frontier.put((aux.score, aux))
        return self

    def get_solution_info(self) -> SolutionInfo:
        if not self.has_solution():
            raise 'A solution must be found to return info!'
        return SolutionInfo(self.solution.get_path_from_root(), self.solution.cost, len(self.visited.keys()),
                            self.frontier.qsize())


class GlobalGreedyAlgorithm(Algorithm):

    def __init__(self, board: Board, player_position: Position, box_positions: List[Position],
                 heuristic: Callable[[Position, List[Position], Board], int]):
        super().__init__(board, player_position, box_positions, heuristic)
        self.frontier = queue.PriorityQueue()

    @staticmethod
    def _visited_value(node: Node) -> int:
        return node.state.heuristic_value

    def _add_to_frontier(self, new_node: Node):
        self.frontier.put((new_node.state.heuristic_value, new_node))

    def _get_item_from_frontier(self) -> Node:
        return self.frontier.get()[1]

    def __iter__(self):
        aux = Node(self.initial_state, 0, None)
        self.frontier.put((aux.state.heuristic_value, aux))
        return self

    def get_solution_info(self) -> SolutionInfo:
        if not self.has_solution():
            raise 'A solution must be found to return info!'
        return SolutionInfo(self.solution.get_path_from_root(), self.solution.cost, len(self.visited.keys()),
                            self.frontier.qsize())


class LocalGreedyAlgorithm(Algorithm):
    def __init__(self, board: Board, player_position: Position, box_positions: List[Position],
                 heuristic: Callable[[Position, List[Position], Board], int]):
        super().__init__(board, player_position, box_positions, heuristic, sort_children=True,
                         sort_children_key=lambda node: node.state.heuristic_value)
        self.frontier = queue.PriorityQueue()

    @staticmethod
    def _visited_value(node: Node) -> int:
        return node.state.heuristic_value

    def _add_to_frontier(self, new_node: Node):
        self.frontier.put((new_node.state.heuristic_value, new_node))

    def _get_item_from_frontier(self) -> Node:
        return self.frontier.get()[1]

    def __iter__(self):
        aux = Node(self.initial_state, 0, None)
        self.frontier.put((aux.state.heuristic_value, aux))
        return self

    def get_solution_info(self) -> SolutionInfo:
        if not self.has_solution():
            raise 'A solution must be found to return info!'
        return SolutionInfo(self.solution.get_path_from_root(), self.solution.cost, len(self.visited.keys()),
                            self.frontier.qsize())

