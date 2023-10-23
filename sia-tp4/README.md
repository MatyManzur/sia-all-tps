# SIA - TP4: Aprendizaje No Supervisado

Autores: Mauro Leandro Báez, Juan Ramiro Castro, Matías Manzur, Amanda Nilsson, Franco David Rupnik, Federico Poyen Shih.
---

Se implementaron los siguientes algoritmos de aprendizaje no supervisado:
 - Kohonen
   - Vecindad y learning rate variable
 - Oja
 - Hopfield

## Ejecución

Para ejecutar el programa, se deben instalar las dependencias usando Pipenv

```bash
pipenv install
```

Luego, en el paquete `plots` se encuentran los scripts para generar los gráficos de los resultados obtenidos. Para ejecutarlos, se debe correr el siguiente comando:

```bash
  pipenv run python3 plots/<script>.py
```

Los programas ejecutables son:
 - [pca_pc1_bar](plots/pca_pc1_bar.py): Genera el grafico de PCA para el dataset.
 - [kohonen_heatmap](plots/kohonen_heatmap.py): Genera el mapa de calor usando la red de Kohonen. Acepta opcionalmente como argumento el path a un archivo json para configurarlo. 
 - [oja_plots](plots/oja_plots.py): Genera los gráficos de Oja. Acepta opcionalmente como argumento el path a un archivo json para configurarlo.
 - [pca_biplot](plots/pca_biplot.py): Genera el biplot de PCA para el dataset.
 - [countries_boxplot](plots/countries_boxplot.py): Genera los boxplot de las variables de los paises, antes y después de estandarizarlas.
 - [kohonen_groups_in_biplot](plots/kohonen_groups_in_biplot): Genera un biplot coloreado con los grupos obtenidos por la red de Kohonen.
 - [mapa](plots/mapa.py): Genera un mapa coloreado con los paises obtenidos por la red de Kohonen.
 - [kohonen_dead_neurons](plots/kohonen_dead_neurons.py): Genera un gráfico de barras con la cantidad de neuronas muertas por cada configuración de la red de Kohonen.

