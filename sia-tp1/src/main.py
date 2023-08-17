import arcade
from algorithm_utils import *
from algorithms import *

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

# Sokoban map (example)
MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

PLAYER_POSITION = Position(7, 4)

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

    # def __search_player__(self):
    #     for y in range(len(self.grid)):
    #         for x in range(len(self.grid[0])):
    #             if self.grid[y][x] == PLAYER:
    #                 self.player_x = x
    #                 self.player_y = y

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
        # draw the grid
        for y, row in enumerate(self.board.map):
            for x, cell in enumerate(row):
                if cell == WALL:
                    arcade.draw_rectangle_filled(x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 (len(self.board.map) - y - 1) * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 SPRITE_SIZE, SPRITE_SIZE,
                                                 arcade.color.BROWN)

        for goal in self.board.goals:
            arcade.draw_rectangle_filled(goal.x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                      (len(self.board.map) - goal.y - 1) * SPRITE_SIZE + SPRITE_SIZE / 2,
                                      SPRITE_SIZE, SPRITE_SIZE,
                                      arcade.color.GREEN)

    def next_state(self):
        if self.algorithm_instance is None:
            raise 'Algorithm not initialized'
        next_node = next(self.algorithm_instance)
        if next_node is None:
            print('No more states!')
            self.finished = True
            return
        self.state = next_node.state

    def start_game(self):
        self.initialized = True
        self.algorithm_instance = iter(self.algorithm)
        self.next_state()


    def on_draw(self):
        if not self.render or not self.initialized or self.finished:
            return
        arcade.draw_circle_filled(self.state.player_position.x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                              self.state.player_position.y * SPRITE_SIZE + SPRITE_SIZE / 2,
                                              SPRITE_SIZE / 2,
                                              arcade.color.YELLOW)
        
        for box in self.state.box_positions:
            arcade.draw_rectangle_filled(box.x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 (len(self.board.map) - box.y - 1)* SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 SPRITE_SIZE, SPRITE_SIZE,
                                                 color = arcade.color.BLUE)

        

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ENTER:
            self.next_state()
        elif key == arcade.key.SPACE and not self.initialized:
            self.start_game()
        # if key == arcade.key.UP:
        #     direction = 'up'
        # elif key == arcade.key.DOWN:
        #     direction = 'down'
        # elif key == arcade.key.LEFT:
        #     direction = 'left'
        # else:
        #     direction = 'right'
        # (result, self.grid) = try_move(self.grid, self.player_x, self.player_y, direction)
        # if result != INVALID:
        #     self.__search_player__()
        # if result == WIN:
        #     print("WIN!")
        #     arcade.set_background_color(arcade.color.GREEN)


def main():
    board = Board(MAP, [Position(2, 1)])
    algorithm = BFSAlgorithm(board, PLAYER_POSITION, [Position(2, 2)])
    game = SokobanGame(board=board, algorithm=algorithm)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()

