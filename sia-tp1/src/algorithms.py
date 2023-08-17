from algorithm_utils import *

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
