from funciones import *
import matplotlib.pyplot as plt
import numpy as np

instagram = leer_archivo('./tests/instagram_famosos_grafo.txt')
mathworld = leer_archivo('./tests/mathworld_grafo.txt')
quince_segundos = leer_archivo('./tests/test_15_segundos.txt')
treinta_segundos = leer_archivo('./tests/test_30_segundos.txt')
aleatorio = leer_archivo('./tests/test_aleatorio.txt')
dos_estrellas = leer_archivo('./tests/test_dosestrellas.txt')

def mDifDens (n):
    d1 = np.random.choice([0.,0.,0.,0.,1.], (n,n))
    d2 = np.random.choice([0.,1.], (n,n))
    d3 = np.random.choice([0.,1.,1.,1.,1.], (n,n))
    
    return d1, d2, d3

def mDifTamyDens (k):
    seq = []
    for i in range (2,k+1,10):
        m1, m2, m3 = mDifDens(i)
        seq += [m1,m2,m3]
    
    return seq

def tamaño (A):
    return A.shape[0]

def densidad (A):
    return np.sum(A)/A.shape[0]

def tiempo (A, p):
    return calcularRanking(A, p)[2]

def grafico_cuantitativo_test(p):
    tamaños = []
    densidades = []
    tiempos = []
    grafos = [instagram,mathworld,quince_segundos,treinta_segundos,aleatorio,dos_estrellas]
    for t in grafos:
        tamaños.append(tamaño(t))
        densidades.append(densidad(t))
        tiempos.append(tiempo(t, p))
        
    c = np.random.rand(len(grafos))
    
    plt.suptitle('ANÁLISIS CUANTITATIVO de TESTS')
    plt.title('Tiempo de procesamiento (s) x Densidad (y tamaño)')
    
    plt.xlabel('Densidad')
    plt.ylabel('Tiempo')
    
    plt.scatter(densidades, tiempos, s=tamaños, c=c, alpha=0.5)
    plt.show()


def grafico_cuantitativo_test_sin_outliers(p):
    tamaños = []
    densidades = []
    tiempos = []
    grafos = [instagram,mathworld,aleatorio,dos_estrellas]
    
    for t in grafos:
        tamaños.append(tamaño(t))
        densidades.append(densidad(t))
        tiempos.append(tiempo(t, p))
     
    nombres = ['Instagram','Mathworld','Aleatorio','Dos estrellas']
   
    
    plt.figure(figsize=(15, 5))
    plt.suptitle('ANÁLISIS CUANTITATIVO de TESTS (sin outlier)')
    
    plt.subplot(121)
    plt.bar(nombres, densidades)
    plt.title('Densidades')
    
    plt.subplot(122)
    c = np.random.rand(len(tamaños)) 
    plt.scatter(tamaños, tiempos, c=c)
    
    
    plt.title('Tiempo de procesamiento (s) x Tamaño')
    plt.show()

def grafico_cuantitativo(k,p):
    tamaños = []
    densidades = []
    tiempos = []
    
    M = mDifTamyDens(k) # Generador matrices con 3 distintas densidades por tamaño.
    for i in M :
        tamaños.append(tamaño(i))
        densidades.append(densidad(i))
        tiempos.append(tiempo(i, p))
        
    c = np.random.rand(len(tamaños)) 
    
    plt.suptitle('ANÁLISIS CUANTITATIVO')
    plt.title('Tiempo de procesamiento (s) x Tamaño (y densidad)')
    plt.xlabel('Tamaño')
    plt.ylabel('Tiempo')
    plt.scatter(tamaños, tiempos, s=densidades, c=c, alpha=0.5)
    plt.show()


