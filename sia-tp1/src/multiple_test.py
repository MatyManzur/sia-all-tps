import json

import algorithms, maps, heuristics, game


def main():
    with (open('../multiple_test_config.json', 'r') as f):
        config = json.load(f)
        tests = config['tests']
        test_results = []
        for test in tests:
            test_map = maps.get_map(test['map'])
            (board, player, boxes) = algorithms.get_positions(test_map)
            algorithm = test['algorithm']
            if algorithm == 'BFS':
                algorithm = algorithms.BFSAlgorithm(board, player, boxes)
            elif algorithm == 'DFS':
                algorithm = algorithms.DFSAlgorithm(board, player, boxes)
            elif algorithm == 'IDDFS':
                algorithm = algorithms.IDDFSAlgorithm(board, player, boxes, test['depth_increment'])
            elif algorithm == 'AStar':
                heuristic = heuristics.get_heuristic(test['heuristic'])
                algorithm = algorithms.AStarAlgorithm(board, player, boxes, heuristic)
            elif algorithm == 'GlobalGreedy':
                heuristic = heuristics.get_heuristic(test['heuristic'])
                algorithm = algorithms.GlobalGreedyAlgorithm(board, player, boxes, heuristic)
            elif algorithm == 'LocalGreedy':
                heuristic = heuristics.get_heuristic(test['heuristic'])
                algorithm = algorithms.LocalGreedyAlgorithm(board, player, boxes, heuristic)
            else:
                raise Exception('Invalid algorithm')
            sokoban = game.SokobanGameNoArcade(board=board, algorithm=algorithm)
            sokoban.run_game()
            print(sokoban.executionInfo)
            test_results.append(sokoban.executionInfo)
        # Writing results in json
        json_object = json.dumps(test_results, indent=4)
        with open(config['output_file'], "w") as outfile:
            outfile.write(json_object)


if __name__ == '__main__':
    main()
