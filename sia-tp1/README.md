# SIA - TP1: Algoritmos de búsqueda

Autores: Mauro Leandro Baez, Juan Ramiro Castro, Matias Manzur, Franco David Rupnik, Federico Poyen Shih.
---

En este proyecto se implementó un motor de juegos en python arcade, una versión simple del juego [Sokoban](https://en.wikipedia.org/wiki/Sokoban) y la implementación de 6 métodos de búsqueda informados y no informados.
- BFS
- DFS
- GlobalGreedy
- LocalGreedy
- A*
- IDDFS

También se implementó 3 heurísticas no triviales
- Manhattan (admisible)
- Precalc (admisible)
- Metro (no admisible)

En el proyecto se implementaron varios casos y mapas del Sokoban de diferentes dificultades.

## Dependencias
El proyecto utiliza varias dependencias denotadas en el Pipfile. Se requiere de Python3 y pip.
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
Para no tocar código python, hay una solución más configurable. test.py levanta de test_config.json el mapa, si renderizar el juego, el tiempo entre renders, el algoritmo, la heurística (si aplicable) y la depth del algoritmo (para IDDFS).
Con esto, un ejemplo del test_config.json esta presente en el repositiorio, pero su formato sería el siguiente:
```json
{
  "map": "TEST_6",
  "render_game": true,
  "render_delay_in_ms": 50,
  "algorithm": "AStar",
  "heuristic": "metro",
  "depth_increment": 100
}
```
Cabe aclarar que render_game estando en true permite poder ver como se resuelve el juego, pero esto hace que el algoritmo se ejecute más lento. Por lo que si se quiere ver el resultado de un algoritmo, se recomienda ponerlo en false.
De la misma manera render_delay_in_ms es el tiempo entre renders, por lo que si se quiere ver con mayor lentitud como se resuelve el juego, se puede aumentar este valor.
### multiple_test.py
Para realizar benchmarks entre algoritmos, multiple_test.py permite levantar la configuración de multiple_test_config.json. 
Este es el formato:
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
Este recibe una lista de algoritmos, mapas, y metadata con que ejecutar, y despues escribe en un json de salida el resultado con este formato:
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
### graph.py
Para ver los resultados, uno puede correr graph.py para ver la comparación de multiple_test_results.
Por lo que será necesario correr mínimamente una vez multiple_test.py con alguna configuración válida.