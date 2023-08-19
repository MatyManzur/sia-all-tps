import arcade
import time
from algorithm_utils import *
from algorithms import *
from maps import *
from heuristics import *

# Constants
SPRITE_SIZE = 30

# Map data
EMPTY = 0
WALL = 1
BOX = 2
TARGET = 3
PLAYER = 4


class SokobanGame(
    arcade.Window
):

    def __init__(self, board: Board, algorithm: Algorithm, render: bool = True):
        super().__init__(len(board.map[0]) * SPRITE_SIZE, len(board.map) * SPRITE_SIZE, "Sokoban Game")
        self.state = None
        self.board = board
        self.render = render
        self.initialized = False
        self.algorithm = algorithm
        self.finished = False

    def setup(self):
        if not self.render:
            return
        arcade.set_background_color(arcade.color.BLACK)
        arcade.start_render()

    def next_state(self):
        if self.algorithm_instance is None:
            raise 'Algorithm not initialized'
        next_node = next(self.algorithm_instance)
        if next_node is None:
            print('No more states!')
            self.finished = True
            return
        self.state = next_node.state
        if self.algorithm.has_finished():
            info = self.algorithm.get_solution_info()
            print("----PATH TO SOLUTION----")
            print(info.path_to_solution)
            for i, state in enumerate(info.path_to_solution):
                print(f"Stage {i}")
                print(f"Player: ({state.player_position.x}, {state.player_position.y})")
                for j, box in enumerate(state.box_positions):
                    print(f"Box {j}: ({box.x}, {box.y})")
                print("------------------------")
            print(f"Cost: {info.final_cost}, "
                  f"Frontier Nodes: {info.frontier_nodes_count}, "
                  f"Expanded Nodes: {info.expanded_nodes_count}")
            print("------------------------")
            self.finished = True

    def start_game(self):
        self.initialized = True
        self.algorithm_instance = iter(self.algorithm)
        self.next_state()

    def draw(self):
        # draw the grid
        arcade.start_render()
        for y, row in enumerate(self.board.map):
            for x, cell in enumerate(row):
                if cell == WALL:
                    arcade.draw_rectangle_filled(x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 y * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 SPRITE_SIZE, SPRITE_SIZE,
                                                 arcade.color.BROWN)
        for goal in self.board.goals:
            arcade.draw_rectangle_filled(goal.x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                         goal.y * SPRITE_SIZE + SPRITE_SIZE / 2,
                                         SPRITE_SIZE, SPRITE_SIZE,
                                         arcade.color.GREEN)
        if not self.render or not self.initialized:
            return
        arcade.draw_circle_filled(self.state.player_position.x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                  self.state.player_position.y * SPRITE_SIZE + SPRITE_SIZE / 2,
                                  SPRITE_SIZE / 2,
                                  arcade.color.YELLOW)

        for box in self.state.box_positions:
            arcade.draw_rectangle_filled(box.x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                         box.y * SPRITE_SIZE + SPRITE_SIZE / 2,
                                         SPRITE_SIZE, SPRITE_SIZE,
                                         color=arcade.color.BLUE)

        arcade.finish_render()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE and not self.initialized:
            self.run_game()

    def run_game(self):
        self.start_game()
        start_time = time.time()
        while not self.finished:
            # time.sleep(20/1000)
            self.next_state()
            if self.render:
                self.draw()
        if self.render:
            arcade.set_background_color(arcade.color.PINK_PEARL)
            self.draw()
        end_time = time.time()
        print(f"Total time elapsed: {end_time - start_time} seconds")


def main():
    (board, player, boxes) = get_positions(MAP_EXAMPLE)
    # algorithm = AStarAlgorithm(data_tuple[0], data_tuple[1], data_tuple[2], trivial_heuristic)
    # algorithm = AStarAlgorithm(data_tuple[0], data_tuple[1], data_tuple[2], manhattan_heuristic)
    # algorithm = AStarAlgorithm(data_tuple[0], data_tuple[1], data_tuple[2], better_manhattan_heuristic)
    # algorithm = GlobalGreedyAlgorithm(data_tuple[0], data_tuple[1], data_tuple[2], manhattan_heuristic)
    # algorithm = LocalGreedyAlgorithm(data_tuple[0], data_tuple[1], data_tuple[2], manhattan_heuristic)
    # algorithm = BFSAlgorithm(data_tuple[0], data_tuple[1], data_tuple[2])
    # algorithm = DFSAlgorithm(board, player, boxes)
    algorithm = IDDFSAlgorithm(board, player, boxes, 1)
    # """
    # pre_calc = PreCalcHeuristic()
    # algorithm = AStarAlgorithm(data_tuple[0], data_tuple[1], data_tuple[2],
                            #    lambda pp, bp, b: pre_calc.pre_calc_heuristic(pp, bp, b))
    # """
    game = SokobanGame(board=board, algorithm=algorithm, render=True)
    game.setup()
    if game.render:
        arcade.run()
    else:
        game.run_game()


if __name__ == "__main__":
    main()
