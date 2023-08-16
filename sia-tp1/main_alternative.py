from typing import Optional, Tuple

import arcade


SCREEN_TITLE = "Sokoban"

CHARACTER_SCALING = 0.5
TILE_SCALING = 0.5

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
EMPTY = 0
WALL = 1
BOX = 2
TARGET = 3
PLAYER = 4
SPRITE_SIZE = arcade.Sprite(":resources:images/tiles/lockRed.png", TILE_SCALING).height
SCREEN_WIDTH = int(len(MAP[0]) * SPRITE_SIZE)
SCREEN_HEIGHT = int(len(MAP) * SPRITE_SIZE)
PLAYER_MOVEMENT_SPEED = 5

class SokobanGame(arcade.Window):
    def __init__(self):
        print(SCREEN_HEIGHT, SCREEN_WIDTH)
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.BROWN)

        self.box_list = None
        self.wall_list = None
        self.target_list = None

        self.player_sprite = None

        self.physics_engine = None

        self.grid = MAP


    def setup(self):
        self.wall_list = arcade.SpriteList(use_spatial_hash=True)
        self.box_list = arcade.SpriteList()
        self.target_list = arcade.SpriteList()

        self.player_sprite = arcade.Sprite(":resources:images/tiles/lockRed.png", TILE_SCALING)


        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                if self.grid[y][x] == WALL:
                    wall = arcade.Sprite(":resources:images/tiles/brickGrey.png", TILE_SCALING)
                    wall.center_x = x * SPRITE_SIZE + SPRITE_SIZE / 2
                    wall.center_y = y * SPRITE_SIZE + SPRITE_SIZE / 2
                    self.wall_list.append(wall)
                elif self.grid[y][x] == BOX:
                    box = arcade.Sprite(":resources:images/tiles/boxCrate.png", TILE_SCALING)
                    box.center_x = x * SPRITE_SIZE + SPRITE_SIZE / 2
                    box.center_y = y * SPRITE_SIZE + SPRITE_SIZE / 2
                    self.box_list.append(box)
                elif self.grid[y][x] == PLAYER:
                    self.player_sprite.center_x = x * SPRITE_SIZE + SPRITE_SIZE / 2
                    self.player_sprite.center_y = y * SPRITE_SIZE + SPRITE_SIZE / 2
                elif self.grid[y][x] == TARGET:
                    target = arcade.Sprite(":resources:images/tiles/lockYellow.png", TILE_SCALING)
                    target.center_x = x * SPRITE_SIZE + SPRITE_SIZE / 2
                    target.center_y = y * SPRITE_SIZE + SPRITE_SIZE / 2
                    self.target_list.append(target)
        self.physics_engine = arcade.PhysicsEngineSimple(self.player_sprite, self.wall_list)

    def on_key_press(self, key, modifiers):
        if key == arcade.key.UP:
            self.player_sprite.change_y = PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.DOWN:
            self.player_sprite.change_y = -PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.LEFT:
            self.player_sprite.change_x = -PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.RIGHT:
            self.player_sprite.change_x = PLAYER_MOVEMENT_SPEED

    def on_key_release(self, symbol: int, modifiers: int):
        if symbol == arcade.key.UP or symbol == arcade.key.DOWN:
            self.player_sprite.change_y = 0
        elif symbol == arcade.key.LEFT or symbol == arcade.key.RIGHT:
            self.player_sprite.change_x = 0

    def on_update(self, delta_time):
        self.physics_engine.update()

        box_hit_list = arcade.check_for_collision_with_list(self.player_sprite, self.box_list)
        for box in box_hit_list:
            box.center_x += self.player_sprite.change_x
            box.center_y += self.player_sprite.change_y


    def on_draw(self):
        arcade.start_render()

        self.player_sprite.draw()
        self.wall_list.draw()
        self.box_list.draw()
        self.target_list.draw()


def main():
    game = SokobanGame()
    game.setup()
    arcade.run()


if __name__ == '__main__':
    main()
