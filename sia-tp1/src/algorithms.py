import queue
from collections import deque
from algorithm_utils import *

class BFSAlgorithm(Algorithm):
    # no hay que pisar next(), solo add_to_frontier y visited_value (o dejar el default),
    # y init() si queremos que no use heuristica
    def __init__(self, board: Board, player_position: Position, box_positions: Set[Position]):
        super().__init__(board, player_position, box_positions,
                         lambda _, __, ___: 0)  # Es desinformado => No usa heuristica
        self.frontier = deque()

    def _add_to_frontier(self, new_node: Node):
        self.frontier.appendleft(new_node)

    def get_algorithm(self):
        return {
            "algorithm": "BFS"
        }


class DFSAlgorithm(Algorithm):

    def __init__(self, board: Board, player_position: Position, box_positions: Set[Position]):
        super().__init__(board, player_position, box_positions,
                         lambda _, __, ___: 0)  # Es desinformado => No usa heuristica
        self.frontier = deque()

    def _add_to_frontier(self, new_node: Node):
        self.frontier.append(new_node)

    def get_algorithm(self):
        return {
            "algorithm": "DFS"
        }


class IDDFSAlgorithm(Algorithm):
    def __init__(self, board: Board, player_position: Position, box_positions: set[Position], depth_increment: int = 1):
        super().__init__(board, player_position, box_positions,
                         lambda _, __, ___: 0, run_get_once=True)  # Es desinformado => No usa heuristica
        self.depth_increment = depth_increment

        self.frontier = deque()

    def __iter__(self):
        self.max_depth = 2
        self.last_frontier = deque()
        super().__iter__()
        return self

    def _add_to_frontier(self, new_node: Node):
        if new_node.depth <= self.max_depth:
            self.frontier.append(new_node)
        else:
            self.last_frontier.append(new_node)

    def _get_item_from_frontier(self) -> Node:
        if len(self.frontier) == 0:
            if len(self.last_frontier) == 0:
                return None
            self.max_depth += self.depth_increment
            # switch
            temp = self.frontier
            self.frontier = self.last_frontier
            self.last_frontier = temp
        return self.frontier.pop()

    def get_algorithm(self):
        return {
            "algorithm": "IDDFS",
            "depth_increment": self.depth_increment
        }



class AStarAlgorithm(Algorithm):

    def __init__(self, board: Board, player_position: Position, box_positions: set[Position],
                 heuristic: Callable[[Position, Set[Position], Board], int]):
        super().__init__(board, player_position, box_positions, heuristic)
        self.frontier = queue.PriorityQueue()
        self.heuristic_name = heuristic.__name__

    @staticmethod
    def _visited_value(node: Node) -> int:
        return node.score

    def _add_to_frontier(self, new_node: Node):
        self.frontier.put((new_node.score, new_node))

    def _get_item_from_frontier(self) -> Node | None:
        try:
            return self.frontier.get(block=False)[1]
        except queue.Empty:
            return None

    def __iter__(self):
        aux = Node(self.initial_state, 0, None)
        self.frontier.put((aux.score, aux))
        return self

    def get_solution_info(self) -> SolutionInfo:
        if not self.has_solution():
            raise 'A solution must be found to return info!'
        return SolutionInfo(self.solution.get_path_from_root(), self.solution.cost, len(self.visited.keys()),
                            self.frontier.qsize())

    def get_algorithm(self):
        return {
            "algorithm": "A*",
            "heuristic": self.heuristic_name
        }


class GlobalGreedyAlgorithm(Algorithm):

    def __init__(self, board: Board, player_position: Position, box_positions: set[Position],
                 heuristic: Callable[[Position, set[Position], Board], int]):
        super().__init__(board, player_position, box_positions, heuristic)
        self.frontier = queue.PriorityQueue()
        self.heuristic_name = heuristic.__name__

    @staticmethod
    def _visited_value(node: Node) -> int:
        return node.state.heuristic_value

    def _add_to_frontier(self, new_node: Node):
        self.frontier.put((new_node.state.heuristic_value, new_node))

    def _get_item_from_frontier(self) -> Node | None:
        try:
            return self.frontier.get(block=False)[1]
        except queue.Empty:
            return None

    def __iter__(self):
        aux = Node(self.initial_state, 0, None)
        self.frontier.put((aux.state.heuristic_value, aux))
        return self

    def get_solution_info(self) -> SolutionInfo:
        if not self.has_solution():
            raise 'A solution must be found to return info!'
        return SolutionInfo(self.solution.get_path_from_root(), self.solution.cost, len(self.visited.keys()),
                            self.frontier.qsize())

    def get_algorithm(self):
        return {
            "algorithm": "GlobalGreedy",
            "heuristic": self.heuristic_name
        }


class LocalGreedyAlgorithm(Algorithm):
    def __init__(self, board: Board, player_position: Position, box_positions: set[Position],
                 heuristic: Callable[[Position, set[Position], Board], int]):
        super().__init__(board, player_position, box_positions, heuristic, sort_children=True,
                         sort_children_key=lambda node: node.state.heuristic_value)
        self.frontier = deque()
        self.heuristic_name = heuristic.__name__

    @staticmethod
    def _visited_value(node: Node) -> int:
        return node.state.heuristic_value

    def _add_to_frontier(self, new_node: Node):
        self.frontier.append(new_node)

    def _get_item_from_frontier(self) -> Node | None:
        try:
            return self.frontier.pop()
        except queue.Empty:
            return None

    def __iter__(self):
        aux = Node(self.initial_state, 0, None)
        self.frontier.append(aux)
        return self

    def get_solution_info(self) -> SolutionInfo:
        if not self.has_solution():
            raise 'A solution must be found to return info!'
        return SolutionInfo(self.solution.get_path_from_root(), self.solution.cost, len(self.visited.keys()),
                            len(self.frontier))

    def get_algorithm(self):
        return {
            "algorithm": "LocalGreedy",
            "heuristic": self.heuristic_name
        }
