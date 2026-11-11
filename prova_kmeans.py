import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math
from pickle import dump, load
import matplotlib.pyplot as plt

# 1. Carregar a base de dados
dados = pd.read_csv('HousingData.csv')

# Tratar valores nulos preenchendo com a mediana
dados = dados.fillna(dados.median())
print("Dados carregados e vazios tratados!")

# 2. Normalizar os dados (Padronização / Z-score)
normalizador = StandardScaler()
modelo_normalizador = normalizador.fit(dados)

# Salvar o modelo normalizador
dump(modelo_normalizador, open('normalizador_boston.model', 'wb'))

# Aplicar a normalização
dados_normalizados = modelo_normalizador.transform(dados)
dados_normalizados = pd.DataFrame(dados_normalizados, columns=dados.columns)
print("Dados normalizados com sucesso!")

# 3. Determinar o número ótimo de clusters (Método do Cotovelo Matemático)
distorcoes = [] 
K = range(1, 15)

for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_normalizados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_normalizados, cluster_model.cluster_centers_, 'euclidean'),
                axis=1) / dados_normalizados.shape[0]
        )
    )

# Calcular a maior distância perpendicular para encontrar o cotovelo
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []

for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distancias.append(numerador/denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print(f"\nO número ótimo de clusters calculado foi: {n_clusters_otimo}")

# 4. Treinar o modelo definitivo e salvar
kmeans_final = KMeans(n_clusters=n_clusters_otimo, random_state=42, n_init=10)
modelo_kmeans = kmeans_final.fit(dados_normalizados)

dump(modelo_kmeans, open('kmeans_boston.model', 'wb'))
print("Modelo definitivo treinado e salvo!")

# ==========================================
# MÓDULO DE INFERÊNCIA
# ==========================================
print("\n--- INICIANDO MÓDULO DE INFERÊNCIA ---")

# 5. Receber dados de uma nova residência
nova_residencia = [[0.02731, 0.0, 7.07, 0.0, 0.469, 6.421, 78.9, 4.9671, 2, 242, 17.8, 396.9, 9.14, 21.6]]

# 6. Carregar os modelos salvos
normalizador_carregado = load(open('normalizador_boston.model', 'rb'))
kmeans_carregado = load(open('kmeans_boston.model', 'rb'))

# 7. Normalizar os dados da nova residência
nova_residencia_normalizada = normalizador_carregado.transform(nova_residencia)

# 8. Inferir a qual grupo a nova residência pertence
grupo_inferido = kmeans_carregado.predict(nova_residencia_normalizada)
print(f"\nA nova residência foi classificada no Grupo: {grupo_inferido[0]}")

# 9. Exibir os dados desnormalizados do centroide inferido
centroide_normalizado = kmeans_carregado.cluster_centers_[grupo_inferido]
centroide_desnormalizado = normalizador_carregado.inverse_transform(centroide_normalizado)

df_centroide = pd.DataFrame(centroide_desnormalizado, columns=dados.columns)
print("\nDados reais (desnormalizados) do Centroide deste grupo:")
print(df_centroide)