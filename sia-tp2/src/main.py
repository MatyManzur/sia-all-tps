import json
from typing import List
from classes import BaseClass


def main():
    with open('config.json') as json_file:
        config = json.load(json_file)


def generate_population(n: int) -> List[BaseClass]:
    return []


if __name__ == '__main__':
    main()
