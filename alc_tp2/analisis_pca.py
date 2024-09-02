import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

from funciones import *

tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
consumidores = pd.read_csv('consumidores_libres.csv', sep = ';')

def normalizar_tabla (dataframe):
    n = dataframe.shape[0]

    # Promedio de todas las filas:
    data_media = np.sum(dataframe, axis=0) / n
    
    # Desviacion estandar de todas las filas:
    data_destand = np.sqrt(np.sum((dataframe - data_media)**2, axis=0) / n)  
       
    # Normalizo:
    data_norm = (dataframe - data_media)/data_destand

    return data_norm, data_media, data_destand

def matriz_convarianza (data_normalizada):
    
    # Calculamos matriz convarianza
    n = data_normalizada.shape[0]
    cov = data_normalizada.T @ data_normalizada / n
    
    # Graficamos
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(cov, cmap= 'BuGn')
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            ax.text(j, i, cov[i, j].round(2), ha="center", va="center", color="black")
    plt.show()

def componentes_principales (dataframe):
    
    # Nos quedamos con la información representativa
    valores = dataframe.iloc[:,2:].values
    
    # Normalizamos los valores
    valores_norm, media, destand = normalizar_tabla(valores)
    
    # Calculamos la matriz de convarianza
    n = valores_norm.shape[0]
    cov = (valores_norm.T @ valores_norm) / n
    
    # Calculamos avals y avecs de la matriz de covarianza
    Avals, Avecs = np.linalg.eigh(cov)
    
    # Ordenamos los autovalores de mayor a menor
    idx = np.argsort (- Avals )
    Avals = Avals[idx]
    Avecs = Avecs[:, idx]
    
    return media, destand, Avals, Avecs

def pca (media, destand, dataframe, autovectores):

    # Nos quedamos con la información representativa
    valores = dataframe.iloc[:,2:].values
    
    # Normalizamos la informacion
    valores = (valores-media)/destand
    
    # Proyectamos sobre autovectores
    valores = autovectores.T @ valores.T
    valores = valores[:3,:].T
    
    # Graficamos
    alimentos = dataframe['Alimento']
    
    fig = plt.figure(figsize = (15,7))
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(valores[:,0], valores[:,1], valores[:,2], alpha=0.5, color='red')
    for xi, yi, zi, texto in zip(valores[:,0], valores[:,1], valores[:,2], alimentos):
        ax.text(xi, yi, zi, texto, size=10)
    plt.title('PCA Tabla nutricional')
    plt.show()
    