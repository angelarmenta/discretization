#developed by Roberto Ángel Meléndez-Armenta
#https://www.youtube.com/@educar-ia

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
file_path = '/Users/angelarmenta/Dropbox/educar-ia/Discretization/Dataset/ds_limones_naranjas.csv'
data = pd.read_csv(file_path)

# Mostrar la información general del dataset
print(data.info())

# Mostrar los primeros 5 registros
print(data.head())

# Discretización por frecuencias iguales para la variable peso
data['peso_discretizado'] = pd.qcut(data['peso'], q=4, labels=False)

# Discretización por K-means para la variable diámetro
kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
data['diametro_discretizado'] = kmeans.fit_predict(data[['diametro']])

# Mostrar los primeros 5 registros del dataset discretizado
print(data.head())

# Guardar el nuevo dataset con las variables discretizadas
data.to_csv('ds_limones_naranjas_discretizado.csv', index=False)
