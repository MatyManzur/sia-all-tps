# SIA - TP1: Algoritmos de búsqueda

Autores: Mauro Leandro Baez, Juan Ramiro Castro, Matias Manzur, Franco David Rupnik, Federico Poyen Shih.
---

En este proyecto se implementó un motor de juegos en python arcade, una versión simple del juego [Sokoban](https://en.wikipedia.org/wiki/Sokoban) y la implementacion de 6 metodos de busqueda informados y no informados.
- BFS
- DFS
- GlobalGreedy
- LocalGreedy
- A*
- IDDFS

Tambien se implemento 3 heuristicas no triviales
- Manhattan (admisible)
- Precalc (admisible)
- Metro (no admisible)

En el proyecto se implementaron varios casos y mapas del Sokoban de diferentes dificultades.

## Dependencias
El proyecto utiliza varias dependencias denotados en el Pipfile. Se requiere de Python3 y pip.
Las dependencias son:
```bash
pip install arcade numpy plotly pandas 
```
Si tiene pipenv, puede correrlo utilizando:
```bash
pipenv install
pipenv shell
```
Y correr los modulos con el comando python.

## Modulos 
Este contiene multiples módulos ejecutables para analizar los diferentes algoritmos.
### game.py
game.py contiene comentado todos los algoritmos para correr, y fue la forma que se fue probando cada algoritmo. SokobanGame recibe el board a correr (definido en maps.py), el algoritmo a correr (algorithms.py) y si renderizar o no.

### test.py
Para no tocar codigo python, hay una solución más configurable. test.py levanta de test_config.json el mapa, si renderizar el juego, el tiempo entre renders, el algoritmo, la heuristica (si aplicable) y la depth del algoritmo (para IDDFS).

### multiple_test.py
Para realizar benchmarks entre algoritmos, multiple_test.py permite levantar la configuracion de multiple_test_config.json. 
Este es el formato
```json
{
  "output_file": "../multiple_test_results.json",
  "tests": [
    {
      "map": "TEST_6",
      "algorithm": "BFS"
    }
  ]
}
```
Este recibe una lista de algoritmos, mapas, y metadata con que ejecutar, y despues escribe en un json de salida el resultado con este formato.
```json
    {
        "algorithm": {
            "algorithm": "BFS"
        },
        "cost": 14,
        "frontier_nodes": 153,
        "expanded_nodes": 355,
        "execution_time": 0.1
    },
```
Para ver los resultados, uno puede correr graph.py para ver la comparacion de multiple_test_results.