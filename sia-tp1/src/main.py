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

INVALID = 0
OK = 1
WIN = 2

# from maps.py
MAP = MAP_EXAMPLE

PLAYER_POSITION = Position(6, 6)


def try_move(grid, x, y, direction):
    new_x = x
    new_y = y
    if direction == 'up':
        new_y += 1
    elif direction == 'down':
        new_y -= 1
    elif direction == 'left':
        new_x -= 1
    elif direction == 'right':
        new_x += 1
    print(f"Moving ({x},{y}) to ({new_x},{new_y})")
    if not (0 <= new_y < len(grid)) or not (0 <= new_x < len(grid[0])):
        print("Invalid move!")
        return INVALID, grid
    if grid[new_y][new_x] in [WALL, PLAYER]:
        print("Invalid move!")
        return INVALID, grid
    result = OK
    if grid[new_y][new_x] == BOX:
        (result, grid) = try_move(grid, new_x, new_y, direction)
        if result == INVALID:
            return INVALID, grid
    if grid[new_y][new_x] == TARGET:
        result = WIN
    grid[new_y][new_x] = grid[y][x]
    grid[y][x] = EMPTY
    return result, grid


class SokobanGame(arcade.Window):

    def __init__(self, board: Board, algorithm: Algorithm, render: bool = True):
        super().__init__(len(board.map[0]) * SPRITE_SIZE, len(board.map) * SPRITE_SIZE, "Sokoban Game")
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
        print(f"Player: ({self.state.player_position.x},{self.state.player_position.y})")
        if self.algorithm.has_finished():
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
            self.start_game()
            while not self.finished:
                time.sleep(20/1000)
                self.next_state()
                self.draw()
            arcade.set_background_color(arcade.color.PINK_PEARL)
            self.draw()


def main():
    board = Board(MAP, [Position(2, 2)])
    algorithm = AStarAlgorithm(board, PLAYER_POSITION, [Position(4, 3)], trivial_heuristic)
    game = SokobanGame(board=board, algorithm=algorithm)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
