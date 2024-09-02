from funciones import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ranking (A,p):
    return calcularRanking(A, p)[0]

def densidad (A):
    return np.sum(A)/A.shape[0]

def tamaño (A):
    return A.shape[0]

def puntaje (A,p):
    return calcularRanking(A, p)[1]

def mejores_rankings (A,n):
    rnks, _, _ = calcularRanking(A,0.99)
    mjs_rnk = np.argsort(rnks)[:n]
    return mjs_rnk

def vars_ranks_distintos_p (A,pp):
    ps = np.arange(1,100,pp)
    ps = ps/100
    rnks = []
    for p in ps:
        rnks.append(ranking(A,p))
        
    rnks_t = np.transpose(rnks) # rnks_t[j] = contiene los puestos de la página j con cada p
    return rnks_t

def vars_scrs_distintos_p (A,pp):
    ps = np.arange(1,100,pp)
    ps = ps/100
    rnks = []
    for p in ps:
        rnks.append(puntaje(A,p))
        
    rnks_t = np.transpose(rnks) # rnks_t[j] = contiene los puestos de la página j con cada p
    return rnks_t

def grafico_cualitativo(A,pp,nrnks):
    dens = densidad(A)
    tamñ = tamaño(A)
    rnks = vars_ranks_distintos_p(A,pp)
    
    fig, ax = plt.subplots(figsize = (12,5))
    ps = np.arange(1,100,pp)/100
    for j in mejores_rankings(A,nrnks):
        ax.plot(ps, rnks[j], label=f"{j}")
    
    ax.set_title(f'ANÁLISIS CUALITATIVO\n Variación de p x Rankings de paginas\n Densidad = {dens}, Tamaño = {tamñ}')
    fig.legend(loc='outside right upper', title = 'Páginas')
    plt.show()

def grafico_cualitativo_scrs(A,pp,nrnks):
    dens = densidad(A)
    tamñ = tamaño(A)
    scrs = vars_scrs_distintos_p(A,pp)
    
    fig, ax = plt.subplots(figsize = (12,5))
    ps = np.arange(1,100,pp)/100
    for j in mejores_rankings(A,nrnks):
        ax.plot(ps, scrs[j], label=f"{j}")
    
    ax.set_title(f'ANÁLISIS CUALITATIVO\n Variación de p x Scores de paginas\n Densidad = {dens}, Tamaño = {tamñ}')
    fig.legend(loc='outside right upper', title = 'Páginas')
    plt.show()
