import pandas as pd

def corregir_tabla_nutricional():
    data = pd.read_csv('tabla_nutricional.csv', sep=';')
    
    # rellenamos valores NaN con 0
    data.fillna(0, inplace=True)
    
    # convertimos las unidades de las columnas de gramos a miligramos
    for columna in data.columns :
        if 'gr' in columna:
            data[columna] *= 1000
            columna_nueva = columna.replace('gr','mg')
            if 'ml' in columna:
                columna_nueva = columna_nueva.replace('ml','µl')
            data.rename(columns={columna: columna_nueva}, inplace=True)
    
    data.to_csv('tabla_nutricional.csv', sep=';', index=False)