import numpy as np
import networkx as nx
from scipy.linalg import solve_triangular
import time
import matplotlib.pyplot as plt
from itertools import product


def leer_archivo(input_file_path):
    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
    	line = f.readline()
    	i = int(line.split()[0]) - 1
    	j = int(line.split()[1]) - 1
    	W[j,i] = 1.0
    f.close()
    
    return W

def dibujarGrafo(W, print_ejes=True):
    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}
    
    N = W.shape[0]
    G = nx.DiGraph(W.T)
    
    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])
    
    nx.draw(G, pos=nx.spring_layout(G), **options)

def ranking_de_scrs (scr): # segun una lista de scores, devuelve el ranking.
    indices_ordenados = np.argsort(scr)[::-1]
    
    rnk = np.zeros(len(indices_ordenados))
    rnk[indices_ordenados] = np.arange(len(indices_ordenados)) 
    
    return rnk

def elim_gaussiana(A): # devuelve la descomposición LU de la matriz A
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return

    for d in range(m-1): # recorre la diagonal
        pivote = Ac[d][d] # elemento de la diagonal es el pivote
        Ac[d+1:,d:d+1] = Ac[d+1:,d:d+1]/pivote # almacena la division de la columna debajo del pivote por el pivote en la misma columna
        Ac[d+1:,d+1:] = Ac[d+1:,d+1:] - Ac[d+1:,d:d+1]@Ac[d:d+1,d+1:] # hace el producto externo de la columna debajo del pivte por la fila a la derecha del pivote y se la resta a la submatriz entre el rango del pivote y el rango de n
        cant_op += 3
            
    L = np.tril(Ac,-1) + np.eye(A.shape[0]) 
    U = np.triu(Ac)
    
    return L, U, cant_op

def matrizD (W): # devuelve la matriz D asociada a W
    n = W.shape[0]
    D = np.zeros((n,n)) # genera una matriz de ceros del mismo tamaño que W
    Wc = W.copy()
    
    cjs = np.sum(Wc, axis = 0) # cjs[k] = #links salientes de la página k en W
    for i in range(len(cjs)):
        if not (cjs[i] == 0):
            cjs[i] = 1/cjs[i] # invierte los elementos de cjs 
    
    np.fill_diagonal(D, cjs) # llena la diagonal con cjs
    return D

def calcularRanking(M, p): # devuelve el ranking, scoring y tiempo de procesamiento de la matriz de un grafo con un valor de p
    stime = time.process_time() # cronometramos el proceso
    
    npages = M.shape[0]
    rnk = np.arange(0, npages) # ind[k] = i, la pagina k tienen el iesimo orden en la lista.
    scr = np.zeros(npages) # scr[k] = alpha, la pagina k tiene un score de alpha 
    R = M @ matrizD(M) 
    e = np.ones(npages)
    
    A = (np.identity(npages) - p*R)
    L, U, _ = elim_gaussiana(A)
    
    y = solve_triangular(L, e, lower = True) # separamos el sistema en dos ecuaciones
    x = solve_triangular(U, y)               #
    
    scr = x/np.linalg.norm(x, ord=1) # normalizamos scr
    rnk = ranking_de_scrs(scr)
    
    ftime = time.process_time() # frenamos el cronometro
    prt = ftime - stime
    
    return rnk, scr, prt 

def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr, prt = calcularRanking(M, p)
    output = np.max(scr)
    
    return output

'''TESTS'''

instagram = leer_archivo('./tests/instagram_famosos_grafo.txt')
mathworld = leer_archivo('./tests/mathworld_grafo.txt')
quince_segundos = leer_archivo('./tests/test_15_segundos.txt')
treinta_segundos = leer_archivo('./tests/test_30_segundos.txt')
aleatorio = leer_archivo('./tests/test_aleatorio.txt')
dos_estrellas = leer_archivo('./tests/test_dosestrellas.txt')

'''NÚMERO DE CONDICION'''

def grafico_condicion(ps): # genera un grafico con la variacion del número de condición de matrices de la forma (I - pWD) para distintos p pasados en una lista
    tamaños = np.arange(5,501,5)
    fig, ax = plt.subplots(figsize = (12,5))
    
    for p in ps: # hacemos un grafico para cada p pasado
        condiciones = []
        for tmñ in tamaños:
            W = generar_matriz_aleatoria(tmñ)
            D = matrizD(W)
            A = np.identity(tmñ) - p*(W@D)
            condiciones.append(np.linalg.cond(A)) # agregamos el n° de condicion a una lista
            
        ax.plot(tamaños, condiciones, label=f"{p}")
         
    fig.legend(loc='outside right upper', title = 'P')
    ax.set_title('Tamaño matrices x Número de condición')

def generar_matriz_aleatoria(n): # genera una matriz de tamaño n con 0 y 1 (con 0 en la diagonal)
    W = np.random.choice([0.,1.],(n,n))
    np.fill_diagonal(W, 0)
    return W

'''ANÁLISIS CUANTITATIVO''' # auxiliares al final del py

def grafico_cuantitativo_test(p): # se pasa como parametro una probabiliad entre 0 y 1 excluyentes
    tamaños = []
    densidades = []
    tiempos = []
    grafos = [instagram,mathworld,quince_segundos,treinta_segundos,aleatorio,dos_estrellas]
    for t in grafos:
        tamaños.append(tamaño(t))
        densidades.append(densidad(t))
        tiempos.append(tiempo(t, p))
        
    c = np.random.rand(len(grafos)) #genera una paleta de colores
    
    plt.suptitle('ANÁLISIS CUANTITATIVO de TESTS')
    plt.title('Tiempo de procesamiento (s) x Densidad (y tamaño)')
    
    plt.xlabel('Densidad')
    plt.ylabel('Tiempo')
    
    plt.scatter(densidades, tiempos, s=tamaños, c=c, alpha=0.5)
    plt.show() # genera un grafico con los parametros pasados (densida,tiempo,tamaño,paleta de colores) 

def grafico_cuantitativo_test_sin_outliers(p): # se pasa como parametro una probabiliad entre 0 y 1 excluyentes
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
    plt.bar(nombres, densidades)#genera un grafico con barras que contempla el nombre de cada pagina y su dencidad
    plt.title('Densidades')
    
    plt.subplot(122)
    c = np.random.rand(len(tamaños))# genera una paleta random de colores
    plt.scatter(tamaños, tiempos, c=c) #genera un grafico del tiempo en funcion del tamaño
    
    
    plt.title('Tiempo de procesamiento (s) x Tamaño')
    plt.show() # genera el grafico

def grafico_cuantitativo(k,p): #recibe como parametro el maximo de los tamaños que quiero generar  y una p que representa la probabilidad 
    tamaños = []
    densidades = []
    tiempos = []
    
    M = mDifTamyDens(k) # Generador matrices con 3 distintas densidades por tamaño.
    for i in M :
        tamaños.append(tamaño(i))
        densidades.append(densidad(i))
        tiempos.append(tiempo(i, p))
        
    c = np.random.rand(len(tamaños)) #genera una paleta de colores
    
    plt.suptitle('ANÁLISIS CUANTITATIVO')
    plt.title('Tiempo de procesamiento (s) x Tamaño (y densidad)')
    plt.xlabel('Tamaño')
    plt.ylabel('Tiempo')
    plt.scatter(tamaños, tiempos, s=densidades, c=c, alpha=0.5)#crea un grafico con los parametros pasados 
    plt.show() #genera el grafico
    
    
'''ANÁLISIS CUALITATIVO''' # auxiliares al final del py

def grafico_cualitativo(A,pp,nrnks): # dado una matriz, un paso de variacion de p, y una cantidad de los mejores rankings 
    dens = densidad(A)
    tamñ = tamaño(A)
    rnks = vars_ranks_distintos_p(A,pp) # rnks[p] = lista de variación del ranking de la pagina p, con un paso de variacion de p = pp
    
    fig, ax = plt.subplots(figsize = (12,5))
    ps = np.arange(1,100,pp)/100 # genera una lista de valores entre 0 y 1 excluyente, con paso pp
    for j in mejores_rankings(A,nrnks):
        ax.plot(ps, rnks[j], label=f"{j}")
    
    ax.set_title(f'ANÁLISIS CUALITATIVO\n Variación de p x Rankings de paginas\n Densidad = {dens}, Tamaño = {tamñ}')
    fig.legend(loc='outside right upper', title = 'Páginas')
    plt.show() # muestra las variaciones del ranking según p de las paginas mejores rankeadas 

def grafico_cualitativo_scrs(A,pp,nrnks): # dado una matriz, un paso de variacion de p, y una cantidad de los mejores rankings
    dens = densidad(A) 
    tamñ = tamaño(A)
    scrs = vars_scrs_distintos_p(A,pp) # scrs[p] = lista de variación del scoring de la pagina p, con un paso de variacion de p = pp
    
    fig, ax = plt.subplots(figsize = (12,5))
    ps = np.arange(1,100,pp)/100 # genera una lista de valores entre 0 y 1 excluyente, con paso pp
    for j in mejores_rankings(A,nrnks):
        ax.plot(ps, scrs[j], label=f"{j}")
    
    ax.set_title(f'ANÁLISIS CUALITATIVO\n Variación de p x Scores de paginas\n Densidad = {dens}, Tamaño = {tamñ}')
    fig.legend(loc='outside right upper', title = 'Páginas')
    plt.show() # muestra las variaciones del scoring según p de las paginas mejores rankeadas
    
'''ANÁLISIS DOS ESTRELLAS'''

def generar_intercalaciones(matriz):
    
    filas = matriz.shape[0] #se fija el tamaño de las filas
    indices = np.arange(filas) #devuelve la cantidad de indices de las filas
    indices_sin_5_y_7 = np.delete(indices, [0,5, 7]) #saca los inices 0 (ya que ninguna pagina se pude apuntar a si mismo) y el indice 5 y 7 (ya que esos indices ya contienen un link entrante en dos estrellas que no quiero que se modifique)
    combinaciones = product([0, 1], repeat=len(indices_sin_5_y_7)) #genera todas las posibles convinaciones de 0 y 1 para los indices sin incluir el 0,5 y 7
    intercalaciones = [] 
    
    for combinacion in combinaciones: #se genera una una iteracion en cada posible combinacion de combinaciones
        matriz_intercalada = matriz.copy() #se genera una copia de la matriz original para que esta no se moifique
        matriz_intercalada[0, indices_sin_5_y_7] = combinacion # Asigna la combinación a la primera fila de la matriz en las columnas especificadas de cada inice sin el 0, el 5 y 7 
        intercalaciones.append(matriz_intercalada) # guara la matriz generada
    
    return intercalaciones # devuelve una lista de todas las posibles matrices que se generaron con combinaciones de la primer fila
    
def minimo_links_dos_estrellas(m, p): #recibe una matriz m y la probabilidad p
    intercalaciones = generar_intercalaciones(m) 
   
    puestos = []
    matrices = []
    puntajes = []
    
   
    for i in range(len(intercalaciones)): #itera en el rango de todas las posibles combinaciones
        puestos.append(calcularRanking(intercalaciones[i], p)[0][0]) #calcula el ranking de la primera pagina para cada intercalacion y la guarda   
        if calcularRanking(intercalaciones[i], p)[0][0] == 0 : #se fija que matrices dejaron a la primer pagina en el primer ranking
            matrices.append(intercalaciones[i]) #guarda la matriz
            
    for i in matrices : #itera sobre las matrices que hacen que la pagina 1 este en el primer ranking
        puntajes.append(np.sum(i[0])) #suma para cada matriz los links entrantes que tiene la primera fila
    return  min(puntajes)-2 #devuelve la minima cantiad de links entrantes y le resta 2 porque el indice 5 y 7 ya tenian un link en dos_estrellas

def grafico_dos_estrellas(m): #grafica la cantidad minima de links en funcion de cada p entre 0 y 1 
     p = np.arange(0.1,1,0.1) #usa p que van de 0.1 a 1 sin incluir 
     links = []
     for i in p:
         links.append(minimo_links_dos_estrellas(m, i))
         
     plt.suptitle('ANÁLISIS DOS ESTRELLAS')
     plt.title('Cantidad minima de links agregados x p')
    
     plt.xlabel('P')
     plt.ylabel('Links')
    
     plt.plot(p, links, c="red")
     plt.show()

'''AUXILIARES'''

def mDifDens (n): # genera 3 matrices de tamaño n con distintas densidades
    d1 = np.random.choice([0.,0.,0.,0.,1.], (n,n))
    d2 = np.random.choice([0.,1.], (n,n))
    d3 = np.random.choice([0.,1.,1.,1.,1.], (n,n))
    
    return d1, d2, d3

def mDifTamyDens (k): #genera matrices de hasta tamaño k 
    seq = []
    for i in range (2,k+1,10):
        m1, m2, m3 = mDifDens(i)
        seq += [m1,m2,m3]
    
    return seq

def ranking (A,p): #devuevle el ranking de la matriz A con p como probabilidad
    return calcularRanking(A, p)[0]

def puntaje (A,p): #devuevle el puntaje de la matriz A con p como probabilidad
    return calcularRanking(A, p)[1]

def tamaño (A): #devuelve el tamaño de matriz cuadrada A
    return A.shape[0]

def densidad (A):
    return np.sum(A)/A.shape[0]

def tiempo (A, p):#devuevle el tiempo de procesamiento de la matriz A con p como probabilidad
    return calcularRanking(A, p)[2]

def mejores_rankings (A,n): # devuelve los indices de las paginas mejores rankeadas para p = 0.99
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
    scrs = []
    for p in ps:
        scrs.append(puntaje(A,p))
        
    scrs_t = np.transpose(scrs) # scrs_t[j] = contiene los puntajes de la página j con cada p
    return scrs_t


