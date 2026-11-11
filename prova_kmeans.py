import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math
from pickle import dump, load
import matplotlib.pyplot as plt

# -----------------------------------------
# TREINAMENTO
# -----------------------------------------

# 1. Carregar base
dados = pd.read_csv('HousingData.csv')

# Preencher nulos com mediana
dados = dados.fillna(dados.median())
print("Dados carregados e tratados!")

# 2. Normalizar separando tipos
# 2.1 Separar categórica (CHAS) das numéricas
dados_num = dados.drop(columns=['CHAS'])
dados_cat = dados['CHAS']

# 2.2 Treinar normalizador só nos números
normalizador = StandardScaler()
modelo_normalizador = normalizador.fit(dados_num)

# Salvar normalizador
dump(modelo_normalizador, open('normalizador_boston.model', 'wb'))

# 2.3 Aplicar normalização
dados_num_norm = modelo_normalizador.transform(dados_num)

# 2.4 Recompor dataframe numérico
dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)

# 2.5 Juntar categórica
dados_normalizados = dados_num_norm.join(dados_cat)
print("Dados normalizados!")

# 3. Descobrir K ótimo (Cotovelo matemático)
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

# Achar a maior distância até a reta
x0, y0 = K[0], distorcoes[0]
xn, yn = K[-1], distorcoes[-1]
distancias = []

for i in range(len(distorcoes)):
    x, y = K[i], distorcoes[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distancias.append(numerador/denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print(f"\nK ótimo calculado: {n_clusters_otimo}")

# 4. Treinar modelo definitivo e salvar
kmeans_final = KMeans(n_clusters=n_clusters_otimo, random_state=42, n_init=10)
modelo_kmeans = kmeans_final.fit(dados_normalizados)

dump(modelo_kmeans, open('kmeans_boston.model', 'wb'))
print("Modelo salvo!")

# -----------------------------------------
# INFERÊNCIA
# -----------------------------------------
print("\n--- INFERÊNCIA ---")

# 5. Receber nova casa
nova_residencia = pd.DataFrame([[0.02731, 0.0, 7.07, 0.0, 0.469, 6.421, 78.9, 4.9671, 2, 242, 17.8, 396.9, 9.14, 21.6]], columns=dados.columns)

# 6. Carregar modelos
normalizador_carregado = load(open('normalizador_boston.model', 'rb'))
kmeans_carregado = load(open('kmeans_boston.model', 'rb'))

# 7. Normalizar separando as colunas
nova_inst_num = nova_residencia.drop(columns=['CHAS'])
nova_inst_cat = nova_residencia['CHAS']

# Normalizar e recompor numéricos
nova_inst_num_norm = normalizador_carregado.transform(nova_inst_num)
nova_inst_num_norm = pd.DataFrame(nova_inst_num_norm, columns=nova_inst_num.columns)

# Juntar categórica
nova_residencia_norm = nova_inst_num_norm.join(nova_inst_cat)

# 8. Inferir grupo
grupo_inferido = kmeans_carregado.predict(nova_residencia_norm)
print(f"\nClassificada no Grupo: {grupo_inferido[0]}")

# 9. Desnormalizar centroide
# 9.1 Pegar centroide padronizado
centroide_norm = kmeans_carregado.cluster_centers_[grupo_inferido]
df_centroide_norm = pd.DataFrame(centroide_norm, columns=dados_normalizados.columns)

# 9.2 Separar colunas
centroide_num = df_centroide_norm.drop(columns=['CHAS'])
centroide_cat = df_centroide_norm['CHAS']

# 9.3 Reverter normalização
centroide_num_desnorm = normalizador_carregado.inverse_transform(centroide_num)
df_centroide_desnorm = pd.DataFrame(centroide_num_desnorm, columns=centroide_num.columns)

# 9.4 Juntar categórica final
df_centroide_final = df_centroide_desnorm.join(centroide_cat)

print("\nCentroide real (desnormalizado):")
print(df_centroide_final)