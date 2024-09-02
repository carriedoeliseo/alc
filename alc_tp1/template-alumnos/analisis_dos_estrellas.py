from funciones import *
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

def generar_intercalaciones(matriz):
    
    filas = matriz.shape[0]
    indices = np.arange(filas)
    indices_sin_5_y_7 = np.delete(indices, [0,5, 7])
    combinaciones = product([0, 1], repeat=len(indices_sin_5_y_7))
    intercalaciones = []
    
    for combinacion in combinaciones:
        matriz_intercalada = matriz.copy()
        matriz_intercalada[0, indices_sin_5_y_7] = combinacion
        intercalaciones.append(matriz_intercalada)
    
    return intercalaciones

    
def dos_estrellas(m, p):
    intercalaciones = generar_intercalaciones(m)
   
    puestos = []
    matrices = []
    puntajes = []
    matrices_ganadoras = []
   
    for i in range(len(intercalaciones)):
        puestos.append(calcularRanking(intercalaciones[i], p)[0][0])  
        if calcularRanking(intercalaciones[i], p)[0][0] == 0 :
            matrices.append(intercalaciones[i])
            
    for i in matrices : 
        puntajes.append(np.sum(i[0]))
        matrices_ganadoras.append(i)
    return  min(puntajes)-2 

def grafico_dos_estrellas(m): 
     p = np.arange(0.1,1,0.1)
     links = []
     for i in p:
         links.append(dos_estrellas(m, i))
         
     c = np.random.rand(len(p))
     plt.suptitle('ANÁLISIS DOS ESTRELLAS')
     plt.title('Cantidad minima de links agregados x p')
    
     plt.xlabel('P')
     plt.ylabel('Links')
    
     plt.plot(p, links, c="red")
     plt.show()