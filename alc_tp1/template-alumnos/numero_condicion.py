from funciones import *
import numpy as np
import matplotlib.pyplot as plt

def grafico_condicion(ps):
    tamaños = np.arange(5,501,5)
    fig, ax = plt.subplots(figsize = (12,5))
    
    for p in ps:
        condiciones = []
        for tmñ in tamaños:
            W = generar_matriz_aleatoria(tmñ)
            D = matrizD(W)
            A = np.identity(tmñ) - p*(W@D)
            condiciones.append(np.linalg.cond(A))
            
        ax.plot(tamaños, condiciones, label=f"{p}")
         
    fig.legend(loc='outside right upper', title = 'P')
    ax.set_title('Tamaño matrices x Número de condición')

def generar_matriz_aleatoria(n):
    W = np.random.choice([0.,1.],(n,n))
    np.fill_diagonal(W, 0)
    return W
