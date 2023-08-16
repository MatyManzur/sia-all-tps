import arcade

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
    [1, 3, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 2, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0, 4, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


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

    def __search_player__(self):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                if self.grid[y][x] == PLAYER:
                    self.player_x = x
                    self.player_y = y

    def __init__(self):
        super().__init__(len(MAP[0]) * SPRITE_SIZE, len(MAP) * SPRITE_SIZE, "Sokoban Game")
        self.grid = MAP

    def setup(self):
        arcade.set_background_color(arcade.color.BLACK)
        self.__search_player__()

    def on_draw(self):
        arcade.start_render()
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == WALL:
                    arcade.draw_rectangle_filled(x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 y * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 SPRITE_SIZE, SPRITE_SIZE,
                                                 arcade.color.BROWN)
                elif cell == BOX:
                    arcade.draw_rectangle_filled(x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 y * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 SPRITE_SIZE, SPRITE_SIZE,
                                                 arcade.color.BLUE)
                elif cell == TARGET:
                    arcade.draw_rectangle_filled(x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 y * SPRITE_SIZE + SPRITE_SIZE / 2,
                                                 SPRITE_SIZE, SPRITE_SIZE,
                                                 arcade.color.GREEN)
                elif cell == PLAYER:
                    arcade.draw_circle_filled(x * SPRITE_SIZE + SPRITE_SIZE / 2,
                                              y * SPRITE_SIZE + SPRITE_SIZE / 2,
                                              SPRITE_SIZE / 2,
                                              arcade.color.YELLOW)

    def on_key_press(self, key, modifiers):
        if key == arcade.key.UP:
            direction = 'up'
        elif key == arcade.key.DOWN:
            direction = 'down'
        elif key == arcade.key.LEFT:
            direction = 'left'
        else:
            direction = 'right'
        (result, self.grid) = try_move(self.grid, self.player_x, self.player_y, direction)
        if result != INVALID:
            self.__search_player__()
        if result == WIN:
            print("WIN!")
            arcade.set_background_color(arcade.color.GREEN)


def main():
    game = SokobanGame()
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
