# Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Datasets
train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')

train.set_index('PassengerId', inplace=True)
test.set_index('PassengerId', inplace=True)

test.shape, train.shape # Diferença entre// train é para treinar o modelo, consequentemente mais colunas e linhas.

test['Survived'] = np.nan

dados = pd.concat([train, test], sort=False)

dados.columns = ['Sobreviveu', 'Classe', 'Nome', 'Sexo', 'Idade', 'IrmaosConjugues', 'PaisFilhos', 'Bilhete',
       'Tarifa', 'Cabine', 'Embarque']
dados['Sexo'].map({'male': 'homem', 'female': 'mulher'})

dados.isnull().sum() # 263 idades nulas! Vamos fazer a predição destas
dados.drop('Cabine', axis=1, inplace=True) # Não nos interessa no momento, mais de mil dados nulos!

# Preenchendo embarque com a moda, e média
dados['Embarque'].unique()
moda = dados['Embarque'].mode()[0] # Nossa moda é 'S'.
dados.fillna({'Embarque': moda}, inplace=True)

media = dados['Tarifa'].mean() # Média é: 33.2955
dados.fillna({'Tarifa': media}, inplace=True)

dados.isnull().sum()

# Verificando as Correlações
dados_numericos = dados.select_dtypes(include=[np.number])
correlacao = dados_numericos.corr()

sns.heatmap(correlacao, annot=True, cmap="OrRd")
plt.show()

dados.drop('Sobreviveu', axis=1, inplace=True) # Aparentemente sem muita correlação com idade, e está faltando bastante dados

# Engenharia de Recursos 1
dados['Nome'].str.extract('([a-zA-Z]+)\.')
dados['Titulo'] = dados['Nome'].str.extract('([a-zA-Z]+)\.')

pd.crosstab(dados['Titulo'], dados['Sexo'])
