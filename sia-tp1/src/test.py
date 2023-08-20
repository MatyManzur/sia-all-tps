import json

import algorithms, maps, heuristics, game


# //TEST3 con iddfs se rompe
# Test10 pre_calc se rompe
def main():
    with (open('../config.json', 'r') as f):
        config = json.load(f)
        test_map = maps.get_map(config['map'])
        (board, player, boxes) = algorithms.get_positions(test_map)
        algorithm = config['algorithm']
        heuristic = heuristics.get_heuristic(config['heuristic'])
        if algorithm == 'BFS':
            algorithm = algorithms.BFSAlgorithm(board, player, boxes)
        elif algorithm == 'DFS':
            algorithm = algorithms.DFSAlgorithm(board, player, boxes)
        elif algorithm == 'IDDFS':
            algorithm = algorithms.IDDFSAlgorithm(board, player, boxes, config['depth_increment'])
        elif algorithm == 'AStar':
            algorithm = algorithms.AStarAlgorithm(board, player, boxes, heuristic)
        elif algorithm == 'GlobalGreedy':
            algorithm = algorithms.GlobalGreedyAlgorithm(board, player, boxes, heuristic)
        elif algorithm == 'LocalGreedy':
            algorithm = algorithms.LocalGreedyAlgorithm(board, player, boxes, heuristic)
        else:
            raise Exception('Invalid algorithm')
        render_game = config['render_game']
        sokoban = game.SokobanGame(board=board, algorithm=algorithm, render=render_game)
        sokoban.setup()
        if sokoban.render:
            sokoban.run()
        else:
            sokoban.run_game()


if __name__ == '__main__':
    main()
