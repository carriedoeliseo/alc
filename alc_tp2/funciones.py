import pandas as pd
import numpy as np
from scipy.linalg import solve_triangular

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
consumidores = pd.read_csv('consumidores_libres.csv', sep = ';')

'''CONSIGNA 1'''

def corregir_tabla_nutricional(dataframe):
    
    # Rellenamos valores NaN con 0
    dataframe.fillna(0, inplace=True)
    
    # Convertimos las unidades de las columnas de miligramos a gramos
    for columna in dataframe.columns :
        if 'mg' in columna:
            dataframe[columna] /= 1000
            columna_nueva = columna.replace('mg','gr')
            dataframe.rename(columns={columna: columna_nueva}, inplace=True)
           
    return dataframe

'''CONSIGNA 2'''

def cumple_margenes_ingesta (dataframe):
    
    # Obtenemos los gramos totales de la canasta básica
    tot_gr = np.sum(dataframe.iloc[:,2:].values)
    
    # Obtenemos los gramos totales de los principales elementos de la dieta
    elem_gr = diccionario_gr_pples_elementos(dataframe)
    paso = True
    
    # Verificamos los margenes de ingesta (cuadro 2)
    if not (tot_gr*0.15 < elem_gr['grasas'] < tot_gr*0.3):
        print('No pasa grasas!, debería contener 15-30%')
        paso = False
    porcentaje = (elem_gr['grasas']/tot_gr)*100
    print(f'Contiene {porcentaje:.2f}% de grasas')
        
    if not  (tot_gr*0.1 < elem_gr['proteinas'] < tot_gr*0.15):
        print('No pasa proteinas!, debería contener 10-15%')
        paso = False
    porcentaje = (elem_gr['proteinas']/tot_gr)*100
    print(f'Contiene {porcentaje:.2f}% de proteinas')
        
    if not (tot_gr*0.55 < elem_gr['hc'] < tot_gr*0.75):
        print('No pasa hidratos!, debería contener 55-75%')
        paso = False
    porcentaje = (elem_gr['hc']/tot_gr)*100
    print(f'Contiene {porcentaje:.2f}% de hidratos de carbono')
        
    gramos = elem_gr['na']
    print(f'Contiene {gramos:.2f}gr de sodio')
        
    if not (25 < elem_gr['fibra']):
        print('No pasa fibra!, debería contener +25 gramos')
        paso = False
    gramos = elem_gr['fibra']   
    print(f'Contiene {gramos:.2f}gr de fibra')
        
    if not (400 < elem_gr['frutas_verduras']):
        print('No pasa frutas y verduras!, debería contener +400 gramos')
        paso = False
    gramos = elem_gr['frutas_verduras']
    print(f'Contiene {gramos:.2f}gr de frutas y verduras')
    return paso 
    
def diccionario_gr_pples_elementos (dataframe):
    
    # rellenamos espacios vacios con 0 y cambiamos todo a gr
    dataframe = corregir_tabla_nutricional(dataframe)

    pples_elementos = ["proteinas", "hc", "grasas", "fibra"]
    gr_totales = {}
    
    # Agregamos a un diccionario la suma de los gr de los pples elementos de la dieta
    for elemento in pples_elementos:
        if esta_elemento(elemento,dataframe):
            suma_gr = np.sum(dataframe[columna_elemento(elemento, dataframe)],axis=0)
            gr_totales[elemento] = suma_gr
            
    # Agregramos la suma de gr de sodio al diccionario
    gr_totales['na'] = np.sum(dataframe.iloc[:,5].values)
    
    # Agregramos la suma de gr de frutas y verduras al diccionario
    gr_totales['frutas_verduras'] = gr_frutas_verduras(dataframe)
    
    return gr_totales   

def esta_elemento (elemento, dataframe):
    columnas = dataframe.columns
    for columna in columnas:
        if elemento in columna.lower():
            return True
    return False
        
def columna_elemento (elemento, dataframe):
    columnas = dataframe.columns
    for columna in columnas:
        if elemento in columna.lower():
            return columna

def gr_frutas_verduras (dataframe):
    frutas_verduras = ['acelga','zanahoria','tomate','lechuga','cebolla','zapallo','manzana','naranja','mandarina' , 'pera' , 'banana']
    cantidad_gr = 0
    
    alimentos = dataframe[dataframe.columns[0]]
    cantidades = dataframe[dataframe.columns[1]]
    for alimento in alimentos:
        if alimento.lower() in (frutas_verduras):
            lista_alimentos = list(alimentos)
            indice = lista_alimentos.index(alimento)
            cantidad_gr += cantidades[indice]
    
    return cantidad_gr

'''CONSIGNA 3'''

def proporciones_por_gramo(dataframe):
    nuevo = dataframe.copy()
    for i in range(nuevo.shape[0]):
        dato = nuevo.iloc[i, 1]  
        for j in range(1, nuevo.shape[1]):
            if dato != 0:
                nuevo.iloc[i, j] /= dato  
    return nuevo

def normalizar_tabla (dataframe):
    n = dataframe.shape[0]

    # Promedio de todas las filas:
    data_media = np.sum(dataframe, axis=0) / n
    
    # Desviacion estandar de todas las filas:
    data_destand = np.sqrt(np.sum((dataframe - data_media)**2, axis=0) / n)  
       
    # Normalizo:
    data_norm = (dataframe - data_media)/data_destand

    return data_norm, data_media, data_destand

def matriz_convarianza (data_normalizada, columnas):
    
    # Calculamos matriz convarianza
    n = data_normalizada.shape[0]
    cov = data_normalizada.T @ data_normalizada / n
    
    # Graficamos
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(cov, cmap= 'BuGn')
    ax.set_xticks(np.arange(len(columnas)), labels=columnas)
    ax.set_yticks(np.arange(len(columnas)), labels=columnas)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            ax.text(j, i, cov[i, j].round(2), ha="center", va="center", color="black")
            
    plt.title('Matriz de convarianza')
    fig.tight_layout()
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

def pca (media, destand, dataframe, autovectores, elev, azim, roll):

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
    ax.set_xlabel('Autovector1')
    ax.set_ylabel('Autovector2')
    ax.set_zlabel('Autovector3')
    fig.tight_layout()
    ax.view_init(elev, azim, roll)
    
def PCA_tabla_nutricional (): # Se recomienda tener el backend grafico seteado en Qt5
    tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla_nutricional = corregir_tabla_nutricional(tabla_nutricional)
    tabla_nutricional = proporciones_por_gramo(tabla_nutricional)
    media, destand, avals, avecs = componentes_principales(tabla_nutricional)
    pca (media, destand, tabla_nutricional, avecs, 0, 0, 0)
    plt.title('PCA Tabla Nutricional')
    plt.show()

'''CONSIGNA 4'''

def consumidores_libres_nutricional(dataframe):
    nuevo = dataframe.drop([1, 2, 3, 4, 5, 9, 12, 13, 14, 18, 20, 23, 24, 25, 29, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55], axis=0)
    nuevo = nuevo.reset_index(drop=True)
    
    nuevo.iloc[[0, 7]] = nuevo.iloc[[7, 0]].values
    nuevo.iloc[[0, 9]] = nuevo.iloc[[9, 0]].values
    nuevo.iloc[[0, 11]] = nuevo.iloc[[11, 0]].values
    nuevo.iloc[[0, 14]] = nuevo.iloc[[14, 0]].values
    nuevo.iloc[[0, 4]] = nuevo.iloc[[4, 0]].values
    nuevo.iloc[[0, 19]] = nuevo.iloc[[19, 0]].values
    nuevo.iloc[[0, 8]] = nuevo.iloc[[8, 0]].values
    nuevo.iloc[[0, 10]] = nuevo.iloc[[10, 0]].values
    nuevo.iloc[[0, 15]] = nuevo.iloc[[15, 0]].values
    nuevo.iloc[[0, 3]] = nuevo.iloc[[3, 0]].values
    nuevo.iloc[[0, 18]] = nuevo.iloc[[18, 0]].values
    nuevo.iloc[[0, 2]] = nuevo.iloc[[2, 0]].values
    nuevo.iloc[[0, 17]] = nuevo.iloc[[17, 0]].values
    nuevo.iloc[[1, 5]] = nuevo.iloc[[5, 1]].values
    nuevo.iloc[[1, 16]] = nuevo.iloc[[16, 1]].values
    nuevo.iloc[[1, 12]] = nuevo.iloc[[12, 1]].values
    nuevo.iloc[[1, 6]] = nuevo.iloc[[6, 1]].values
    nuevo.iloc[[1, 13]] = nuevo.iloc[[13, 1]].values
    
    return nuevo

def PCA_consumidores_libres (): # Se recomienda tener el backend grafico seteado en Qt5
    tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla_nutricional = corregir_tabla_nutricional(tabla_nutricional)
    tabla_nutricional = proporciones_por_gramo(tabla_nutricional)
    media, destand, avals, avecs = componentes_principales(tabla_nutricional)
    consumidores_nutricional = consumidores_libres_nutricional(tabla_nutricional)
    pca(media, destand, consumidores_nutricional, avecs, 0, 0, 0)
    plt.title('PCA Consumidores Libres')
    plt.show()

'''CONSIGNA 5'''

def separar_macros(dataframe):
    datos = pd.DataFrame()
    tabla = proporciones_por_gramo(dataframe)
    datos = tabla.iloc[:, :5]
    return datos

def tablas_macros(data, macros):
    f, c = data.shape
    m = macros.shape[1]
    
    precio_hc = data.copy()
    precio_hc.iloc[:, 1:] = precio_hc.iloc[:, 1:].astype('float64')
    precio_proteina = data.copy()
    precio_proteina.iloc[:, 1:] = precio_proteina.iloc[:, 1:].astype('float64')
    precio_grasas = data.copy()
    precio_grasas.iloc[:, 1:] = precio_grasas.iloc[:, 1:].astype('float64')
    
    for fila in range (f):
        for macro in range(2,m):
            if macro == 2:
                nutriente = macros.iloc[fila,macro]
                if nutriente != 0:
                    precio_hc.iloc[fila,2:] = precio_hc.iloc[fila,2:].values / nutriente
                else:
                    precio_hc.iloc[fila,2:] = precio_hc.iloc[fila,2:].values * nutriente
            elif macro == 3:
                nutriente = macros.iloc[fila,macro]
                if nutriente != 0:
                    precio_proteina.iloc[fila,2:] = precio_proteina.iloc[fila,2:].values / nutriente
                else:
                    precio_proteina.iloc[fila,2:] = precio_proteina.iloc[fila,2:].values * nutriente
            elif macro == 4:
                nutriente = macros.iloc[fila,macro]
                if nutriente != 0:
                    precio_grasas.iloc[fila,2:] = precio_grasas.iloc[fila,2:].values / nutriente
                else:
                    precio_grasas.iloc[fila,2:] = precio_grasas.iloc[fila,2:].values * nutriente
                    
    return precio_hc, precio_proteina, precio_grasas

def eliminar_cantidades_nulas(dataframe):
    nuevo = dataframe.copy()
    for i in range(dataframe.shape[0]):
        dato = dataframe.iloc[i, 1]  
        if dato == 0 : 
            nuevo = nuevo.drop(i,axis = 0 )
    nuevo = nuevo.reset_index(drop=True)      
    return nuevo

def ejes (tabla):
    m, n = tabla.shape
    xs = np.arange(0,n,1)
    xs = np.tile(xs, m)
    copia = tabla.copy()
    ys = np.reshape(copia,copia.size)

    return xs, ys

def M_polinomial (xs, grado):
    n = xs.size
    res = np.empty((n,grado+1))
    for k in range (grado+1):
        columna_k = xs**k
        res[:,k] = columna_k
    return res

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return

    for d in range(m-1):
        pivote = Ac[d][d]
        Ac[d+1:,d:d+1] = Ac[d+1:,d:d+1]/pivote
        Ac[d+1:,d+1:] = Ac[d+1:,d+1:] - Ac[d+1:,d:d+1]@Ac[d:d+1,d+1:]
        cant_op += 3
            
    L = np.tril(Ac,-1) + np.eye(A.shape[0]) 
    U = np.triu(Ac)
    
    return L, U, cant_op

def solve_luxb (L,U,b):
    y = solve_triangular(L, b, lower = True)
    x = solve_triangular(U, y)
    
    return y, x
    
def evaluacion_polinomial (coeficientes, x):
    grado = coeficientes.size-1
    res = 0
    for k in range (grado+1):
        coef = coeficientes[k]
        res += coef * (x**k)
    return res    

def generador_puntos (coeficientes, xs):
    num_puntos = xs.size
    puntos = np.empty((num_puntos,2))
    for i in range (num_puntos):
        puntos[i][0] = xs[i]
        puntos[i][1] = evaluacion_polinomial(coeficientes, xs[i])
    return puntos

def minimos_cuadrados_graf(tabla, grado, color):
    
    
    xs, ys = ejes(tabla)
    
    
    M = M_polinomial(xs,grado)
    MtM = M.T @ M
    Mty = M.T @ ys
    L, U, _ = elim_gaussiana(MtM)
    
    
    y, coeficientes = solve_luxb(L, U, Mty)
    puntos_polinomio = generador_puntos(coeficientes, xs)
    
    
    tope = np.unique(puntos_polinomio[:,0]).size
    mc = puntos_polinomio[:tope]
    
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c='gray')
    ax.plot(mc[:,0], mc[:,1], c=color)
    
    plt.xlabel('Meses')
    plt.ylabel('Precio x gramo')
    plt.title(f'Pendiente:{coeficientes[1]:.2f}, Precio Inicial:{mc[0,1]:.2f}')
    
def minimos_cuadrados(tabla, grado):
    
    
    xs, ys = ejes(tabla)
    
    
    M = M_polinomial(xs,grado)
    MtM = M.T @ M
    Mty = M.T @ ys
    L, U, _ = elim_gaussiana(MtM)
    
    
    y, coeficientes = solve_luxb(L, U, Mty)
    puntos_polinomio = generador_puntos(coeficientes, xs)
    
    
    tope = np.unique(puntos_polinomio[:,0]).size
    return puntos_polinomio[:tope]

def diferencia_porcentajes(lista):
    diferencias = []
    for i in range(1,len(lista)):
        valor_anterior = lista[i-1]
        valor_actual = lista[i]
        diferencia = ((valor_actual - valor_anterior)/valor_anterior) * 100
        diferencias.append(diferencia)
    return diferencias

def tabla_aumentos_macros_porcentual (hc, proteinas, grasas):
    tabla_aumentos = np.empty((3,4))
    dataframe = pd.DataFrame()
    
    aumentos_carbohidratos = hc[:,1]
    porcentajes_precios_carbohidratos = diferencia_porcentajes(aumentos_carbohidratos)
       
    aumentos_grasas = proteinas[:,1]
    porcentajes_precios_grasas = diferencia_porcentajes(aumentos_grasas)
        
    aumentos_proteina = grasas[:,1]
    porcentajes_precios_proteina = diferencia_porcentajes(aumentos_proteina)
    
    tabla_aumentos[0] = porcentajes_precios_carbohidratos
    tabla_aumentos[1] = porcentajes_precios_proteina
    tabla_aumentos[2] = porcentajes_precios_grasas 
    
    dataframe["Nutrientes"] = ["Carbohidratos","Proteinas","Grasas"]
    dataframe["Mes 1 (%)"]= tabla_aumentos[:,0]
    dataframe["Mes 2 (%)"]= tabla_aumentos[:,1]
    dataframe["Mes 3 (%)"]= tabla_aumentos[:,2]
    dataframe["Mes 4 (%)"]= tabla_aumentos[:,3]
    
    return dataframe

def tabla_aumentos_macros_precios (hc, proteinas, grasas):
    tabla_aumentos = np.empty((3,4))
    dataframe = pd.DataFrame()
    
    aumentos_carbohidratos = hc[:,1]
    aumentos_grasas = proteinas[:,1]
    aumentos_proteina = grasas[:,1]
    
    s = aumentos_carbohidratos.size
    
    for i in range (s-1, 0, -1):
        aumentos_carbohidratos[i] = aumentos_carbohidratos[i]-aumentos_carbohidratos[i-1]
        aumentos_grasas[i] = aumentos_grasas[i]-aumentos_grasas[i-1]
        aumentos_proteina[i] = aumentos_proteina[i]-aumentos_proteina[i-1]
    
    tabla_aumentos[0] = aumentos_carbohidratos[1:]
    tabla_aumentos[1] = aumentos_grasas[1:]
    tabla_aumentos[2] = aumentos_proteina[1:]
    
    dataframe["Nutrientes"] = ["Carbohidratos","Proteinas","Grasas"]
    dataframe["Mes 1 ($ x gr)"]= tabla_aumentos[:,0]
    dataframe["Mes 2 ($ x gr)"]= tabla_aumentos[:,1]
    dataframe["Mes 3 ($ x gr)"]= tabla_aumentos[:,2]
    dataframe["Mes 4 ($ x gr)"]= tabla_aumentos[:,3]
    
    return dataframe
    
''' CONSIGNA 6'''

def variacion_precios(data):
    tabla = data.iloc[:,2:].values
    aumentos = minimos_cuadrados(tabla, 1)[:,1]
    return aumentos

def tabla_aumentos_rubros(carnes, almacen, verduleria):

    tabla_porcentajes = np.empty((3,4))

    aumentos_carnes = variacion_precios(carnes)
    aumentos_almacen = variacion_precios(almacen)
    aumentos_verduleria = variacion_precios(verduleria)

    porcentajes_aumentos_carnes = diferencia_porcentajes(aumentos_carnes)
    porcentajes_aumentos_almacen = diferencia_porcentajes(aumentos_almacen)
    porcentajes_aumentos_verduleria = diferencia_porcentajes(aumentos_verduleria)

    tabla_porcentajes[0] = porcentajes_aumentos_carnes 
    tabla_porcentajes[1] = porcentajes_aumentos_almacen
    tabla_porcentajes[2] = porcentajes_aumentos_verduleria

    data = pd.DataFrame()
    data['Rubro'] = ['Carne','Almacén', 'Verdulería']
    data["Mes 1"]= tabla_porcentajes[:,0]
    data["Mes 2"]= tabla_porcentajes[:,1]
    data["Mes 3"]= tabla_porcentajes[:,2]
    data["Mes 4"]= tabla_porcentajes[:,3]

    return data

def disminuir_carne(dataframe,porcentaje):
    tabla = dataframe.copy()
    for i in range(7,15):
        tabla.iloc[i,1:] = tabla.iloc[i,1:] * porcentaje
    tabla.iloc[37,1:] = tabla.iloc[37,1:] * porcentaje
    tabla.iloc[38,1:] = tabla.iloc[38,1:] * porcentaje
    return tabla

'''CONSIGNA 7'''

def diferencia_precio_carnes_consumidores ():
    tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla_nutricional = corregir_tabla_nutricional(tabla_nutricional)
    consumidores_nutricional = consumidores_libres_nutricional(tabla_nutricional)
    
    consumidores = pd.read_csv('consumidores_libres.csv', sep = ';')
    consumidores = proporciones_por_gramo(consumidores)
    consumidores = consumidores.drop(10,axis=0)
    consumidores = consumidores.reset_index(drop=True)
    
    precio_inicial = 0
    precio_final = 0
    
    for f in range (16,20,1):
        precio_inicial += consumidores_nutricional.iloc[f,1]*consumidores.iloc[f,2]
        precio_final += consumidores_nutricional.iloc[f,1]*consumidores.iloc[f,6]
        
    return precio_final - precio_inicial

def gramos_totales_carne_sin_precio ():
    tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla_nutricional = corregir_tabla_nutricional(tabla_nutricional)
    
    gr = 0
    for i in [9,12,13,14,37,38]:
        gr += tabla_nutricional.iloc[i,1]
        
    return gr
    
def cantidad_para_comprar (precio):
    consumidores = pd.read_csv('consumidores_libres.csv', sep = ';')
    consumidores = proporciones_por_gramo(consumidores)
    consumidores = consumidores.drop(10,axis=0)
    consumidores = consumidores.reset_index(drop=True)
    valores = consumidores.iloc[:16,6].values
    
    cantidades = pd.DataFrame()
    
    for f in range (valores.size):
        valores[f] = precio / valores[f]
    
    cantidades['Alimento'] = consumidores.iloc[:16,0].values
    cantidades[f'Gramos para comprar con ${precio}'] = valores
    
    return cantidades

def tabla_suplementada_huevo_y_pan (gr_huevo_disponible, gr_pan_disponible, porcentaje_huevo, porcentaje_pan):
    tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla_nutricional = corregir_tabla_nutricional(tabla_nutricional)
    tabla_nutricional = disminuir_carne(tabla_nutricional, 0.815)
    
    aumento_huevo = (gr_huevo_disponible*porcentaje_huevo / tabla_nutricional.iloc[0,1]) + 1
    aumento_pan = (gr_pan_disponible*porcentaje_pan / tabla_nutricional.iloc[15,1]) + 1
    
    tabla_nutricional.iloc[0,1:] = tabla_nutricional.iloc[0,1:].values*aumento_huevo
    tabla_nutricional.iloc[26,1:] = tabla_nutricional.iloc[26,1:].values*aumento_pan
    
    return tabla_nutricional

def cumple_ingesta_suplementado_huevo_y_pan():
    tabla = tabla_suplementada_huevo_y_pan ()
    return cumple_margenes_ingesta(tabla)


'''CONSIGNA 8'''
#Veamos cuanto gramos menos de proteina nos genera la disminucion del 18,5% del consumo de carne

def cantidad_proteina():
    tabla = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla = corregir_tabla_nutricional(tabla)
    
    # Sumamos todas las proteinas
    proteinas = tabla["Proteinas (gr)"].sum()
    
    return proteinas

def cantidad_proteinas_con_disminucion_carnes():
    tabla = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla = corregir_tabla_nutricional(tabla)
    
    # Disminuimos el consumo de carne un 18.5%
    tabla = disminuir_carne(tabla, 0.815)
    
    # Sumamos todas las proteinas
    proteinas = tabla["Proteinas (gr)"].sum()
    
    return proteinas
    
# buscamos puntos alejados en cuanto a distancia eucledeana en grafico de tabla 1

def pca_2d ():
    tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla_nutricional = corregir_tabla_nutricional(tabla_nutricional)
    tabla_nutricional = proporciones_por_gramo(tabla_nutricional)

    media, destand, avals, avecs = componentes_principales(tabla_nutricional)
    valores = tabla_nutricional.iloc[:,2:].values
    valores = (valores-media)/destand
    valores = avecs.T @ valores.T
    valores = valores[:2,:].T

    alimentos = tabla_nutricional['Alimento']
    fig, ax = plt.subplots(figsize = (10,7))

    ax.scatter(valores[:,0], valores[:,1], alpha=0.5, color='red')
    for xi, yi ,texto in zip(valores[:,0], valores[:,1],alimentos):
        ax.text(xi, yi,texto, size=10)
    plt.title('PCA Tabla nutricional (2d)')
    plt.xlabel('Autovector 1')
    plt.ylabel('Autovector 2')
    plt.show()
    
# En el grafico se puede ver que el arroz esta a distancia eucledeana conciderable de las carnes
# vemos que pasa si agregamos la cantidad de arroz necesaria para aquiparar las proteinas que perdimos

# notar que se perdieron  cantidad_proteina() - cantidad_proteinas_con_disminucion_carnes() es de 8.33 gramos de proteinas
# veamos cuanta es la cantidad de arroz que nos conviene meter en gramos para quiparar la proteina perdida

def cantidad_dulcebatata_necesaria():
    tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla_nutricional = corregir_tabla_nutricional(tabla_nutricional)
    tabla_nutricional = proporciones_por_gramo(tabla_nutricional)
    
    # Obtenemos la proteina en 1 gramo de mermelada
    proteina_dulcebatata = tabla_nutricional.iloc[43,3]
    cantidad_proteina_faltante = cantidad_proteina() - cantidad_proteinas_con_disminucion_carnes()
    
    # Devolvemos la cantidad de arroz necesaria para cubrir el deficit de proteina
    return cantidad_proteina_faltante / proteina_dulcebatata
    
# veamos que pasa con la tabla si sumamos 980 gramos de dulce de batata a la tabla y aumentamos las proporciones correspondientes 

def tabla_suplementada_dulcebatata():
    
    # Obtenemos la tabla con disminucion de carne
    tabla = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla = corregir_tabla_nutricional(tabla)
    tabla = disminuir_carne(tabla, 0.815)
    
    # Obtenemos cantidad de dulce de batata actual y lo necesario para cubrir el déficit de proteina
    cantidad_arroz = tabla.iloc[43,1]
    aumento = cantidad_dulcebatata_necesaria()
    
    # Aumentamos el dulce de batata para cubrir la proteína
    porcentaje_de_aumento = (aumento / cantidad_arroz) + 1 
    tabla.iloc[43,1:] = tabla.iloc[43,1:] * porcentaje_de_aumento
    proteinas = tabla["Proteinas (gr)"].sum()
    
    return tabla, proteinas

#ahora vemos si esta nueva tabla cumple los valores de la oms 

def cumple_ingesta_suplementado_dulcebatata():
    tabla, _ = tabla_suplementada_dulcebatata()
    return cumple_margenes_ingesta(tabla)

# como otra opcion vamos a agarrar queso de rallar ya que se encuentra bastante alejado pero contieneuna minima cantidad de proteina
# veamos que pasa si sustituimos la proteina faltante de la carne con queso de rallar

def cantidad_queso_necesaria():
    tabla_nutricional = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla_nutricional = corregir_tabla_nutricional(tabla_nutricional)
    tabla_nutricional = proporciones_por_gramo(tabla_nutricional)
    
    # Obtenemos la proteina en 1 gramo de queso de rallar
    proteina_queso = tabla_nutricional.iloc[5,3]
    cantidad_proteina_faltante = cantidad_proteina() - cantidad_proteinas_con_disminucion_carnes()
    
    # Devolvemos la cantidad de queso necesaria para cubrir el deficit de proteina
    return cantidad_proteina_faltante / proteina_queso

# veamos que pasa con la tabla si sumamos 25,4 gramos de queso a la tabla y aumentamos las proporciones correspondientes
 
def tabla_suplementada_queso():
    
    # Obtenemos la tabla con disminucion de carne
    tabla = pd.read_csv('tabla_nutricional.csv', sep = ';')
    tabla = corregir_tabla_nutricional(tabla)
    tabla = disminuir_carne(tabla,0.815)
    
    # Obtenemos cantidad de queso de rallar actual y lo necesario para cubrir el déficit de proteina
    cantidad_queso = tabla_nutricional.iloc[5,1] 
    aumento = cantidad_queso_necesaria()
    
    # Aumentamos el queso para cubrir la proteína
    porcentaje_de_aumento = (aumento / cantidad_queso) + 1
    tabla.iloc[5,1:] = tabla.iloc[5,1:] * porcentaje_de_aumento
    proteinas = tabla["Proteinas (gr)"].sum()
    return tabla, proteinas    
    
#ahora vemos si esta nueva tabla cumple los valores de la oms
 
def cumple_ingesta_suplementado_queso():
    tabla, _ = tabla_suplementada_queso()
    return cumple_margenes_ingesta(tabla)    
    

#Podemos ver que en ninguno de los dos casos pasan los valores propuestos por la oms    
    
    
























        
    