import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Carregando os dados
dados = np.loadtxt('new_diabetes.csv', delimiter=',', skiprows=1, usecols=( 0, 1, 2, 3, 4, 5, 6,7))  
# Substitua 'diabetes.csv' pelo nome do seu arquivo de dados
# Selecione as colunas relevantes para o SOM (neste exemplo, colunas 1, 2, 5 e 6)

# Normalizando os dados
dados = (dados - np.min(dados, axis=0)) / (np.max(dados, axis=0) - np.min(dados, axis=0))

# Definindo os hiperparâmetros do SOM
dim_entrada = dados.shape[1]
dim_saida = 5
num_epocas = 5
taxa_aprendizado = 0.1

# Criando e treinando o SOM
som = MiniSom(dim_saida, dim_saida, dim_entrada, sigma=1.0, learning_rate=taxa_aprendizado)
som.random_weights_init(dados)
som.train_batch(dados, num_epocas)

# Visualizando os resultados
plt.figure(figsize=(dim_saida, dim_saida))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Mapa de distância
plt.colorbar()

# Marcando as amostras com seus rótulos (considerando que exista uma coluna com os rótulos no arquivo de dados)
rotulos = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1, usecols=-1, dtype=str)  
# Substitua 'diabetes.csv' pelo nome do seu arquivo de dados e selecione a coluna com os rótulos
marcadores = ['o', 's']  # Marcadores para cada classe (por exemplo, 'o' para negativo e 's' para positivo)

for i, x in enumerate(dados):
    bmu = som.winner(x)  # Encontrar o BMU para cada amostra
    plt.plot(bmu[0] + 0.5, bmu[1] + 0.5, marcadores[int(rotulos[i])], markeredgecolor='k', markerfacecolor='None', markersize=12, markeredgewidth=2)

plt.xticks(np.arange(dim_saida + 1))
plt.yticks(np.arange(dim_saida + 1))
plt.grid()
plt.show()
