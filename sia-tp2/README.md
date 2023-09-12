# SIA - TP2: Algoritmos genéticos

Autores: Mauro Leandro Baez, Juan Ramiro Castro, Matias Manzur, Franco David Rupnik, Federico Poyen Shih.
---

En este proyecto se implemento un motor de algoritmos genéticos para resolver un problema de optimización sobre clases en un "RPG".

## Implementación

- Cruza
  - Cruce de un punto
  - Cruce de dos puntos
  - Cruce uniforme
  - Cruce anular
- Mutación (uniforme y no uniforme)
  - Mutación de Gen
  - Mutación MultGen
- Selección
  - Elite
  - Ruleta Universal
  - Boltzmann
  - Torneos (Deterministico y Probabilistico)
  - Ranking
- Reemplazo
  - Young
  - Tradicional
- Criterios de Corte
  - Cantidad de generaciones
  - Estructura
  - Contenido
  - Aptitud

## Dependencias

El proyecto utiliza varias dependencias denotadas en el Pipfile. Se requiere de Python3 y pip.

```bash
pip install numpy pandas plotly
```

En caso de tener pipenv instalado, pueden correrlo utilizando:

```bash
pipenv install
pipenv shell
```

Para correr el programa, basta con utilizar el siguiente comando, pasando obligatoriamente por primer argumento el archivo de configuración (basado en los ejemplos en el directorio `config`) y como segundo argumento opcional el archivo JSON donde se guardara la salida del programa:

```bash
python main.py config/config.json output.json
```

En el archivo de output se encuentra la información de todos los miembros de cada generación, impresos de mayor a menor fitness.


