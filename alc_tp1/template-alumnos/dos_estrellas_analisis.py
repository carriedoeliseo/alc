from funciones import *
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

dos_estrellas = leer_archivo('./tests/test_dosestrellas.txt')
    
def buscarCoordsCeros (W):
    ceros = []
    
    for i in range (W.shape[0]):
        for j in range (W.shape[1]):
            if (W[i][j] == 0) and (i != j): 
                ceros.append((i,j))
            
    return ceros

def buscarCoordsCerosEnPrimeraFila (W):
    ceros = []
    
    for j in range (W.shape[1]):
            if (W[0][j] == 0) and (j != 0): 
                ceros.append((0,j))
             
    return ceros
        
def subSeqCeros (ceros, largo):
    return [list(subseq) for subseq in combinations(ceros, largo)]

def probarRankingConSubseqs (W, l):
    dos_estrellas = leer_archivo('./tests/test_dosestrellas.txt')
    ps = np.arange(1,100,1)/100
    
    ceros = buscarCoordsCeros(W)
    subseqs = subSeqCeros(ceros, l)
    
    for subseq in subseqs:
        W = dos_estrellas.copy()
        
        for coord in subseq:
            W[coord[0]][coord[1]] = 1
        
        for p in ps:
            _, scr, _ = calcularRanking(W,p)
            if np.max(scr) == scr[0]:
                return subseq, p
        
    return 'Pág 1 no es la mejor'

def probarRankingConSubseqs_LinksAlUno (W, l):
    dos_estrellas = leer_archivo('./tests/test_dosestrellas.txt')
    ps = np.arange(1,100,l)/100
    
    ceros = buscarCoordsCerosEnPrimeraFila(W)
    subseqs = subSeqCeros(ceros, l)
    
    for subseq in subseqs:
        W = dos_estrellas.copy()
        
        for coord in subseq:
            W[coord[0]][coord[1]] = 1
        
        for p in ps:
            _, scr, _ = calcularRanking(W,p)
            if np.max(scr) == scr[0]:
                return subseq, p
        
    return 'Pág 1 no es la mejor'
