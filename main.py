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
dados['Sexo'] = dados['Sexo'].map({'male': 'homem', 'female': 'mulher'})

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
# plt.show()

dados.drop('Sobreviveu', axis=1, inplace=True) # Aparentemente sem muita correlação com idade, e está faltando bastante dados

# Engenharia de Recursos 1
dados['Nome'].str.extract('([a-zA-Z]+)\.')
dados['Titulo'] = dados['Nome'].str.extract('([a-zA-Z]+)\.')

pd.crosstab(dados['Titulo'], dados['Sexo'])

dados['Titulo'] = dados['Titulo'].apply(lambda x: 'Outros' if x not in ['Miss', 'Master', 'Mr', 'Mrs'] else x)
dados.loc[dados['Titulo'] == 'Master'] # Verificação de que o Titulo 'Master' se refere a 'Meninos'. Idade mais alta 14. Média 5 anos.

traducao_titulos = {
    'Master' : 'Meninos',
    'Miss': 'Solteira',
    'Mr': 'Homem adulto',
    'Mrs': 'Casada'
}

dados['Titulo'] = dados['Titulo'].map(traducao_titulos)

# Averiguar informações do DataFrame
dados.info()

## idade float devido a idade de crianças, que algumas colocavam até os meses
## para fazer a predição de idades aqui vemos que não vamos usar: Nome e Bilhete

dados.drop(['Nome', 'Bilhete'], axis =1, inplace = True)

# Engenharia de Recursos 2
## estudar a possibilidade  de identificar 'meninas' similar aos 'meninos'
## através da combinação "solteira" e "esta com os pais"

solteira_comPais = dados.loc[(dados['Titulo'] == 'Solteira') & (dados['PaisFilhos']>= 1)]
solteira_comPais['Idade'].mean()

plt.hist(solteira_comPais['Idade'], bins=15) # Verificando a distribuição da idade
plt.show()

dados.loc[(dados['Titulo'] == 'Solteira')]['Idade'].mean() # Confirmando nossa engenharia de recursos
# solteira_comPais tem média 12.1788, enquanto a média calculada acima de mulheres solteiras é de 21.7742

plt.hist(dados.loc[(dados['Titulo'] == 'Solteira')]['Idade'], bins=15, color='blue')
plt.show() # Distribuição idade mulheres solteiras
 
dados.loc[(dados['Titulo'] == 'Casada')]['Idade'].mean() # Como confirmamos, casadas média 39.994
# por curiosidade, olhar a distribuição
plt.hist(dados.loc[(dados['Titulo'] == 'Casada')]['Idade'], bins=15, color='blue')
plt.show()

solteira_comPais.index
dados['solteira_com_pais'] = 0

for idx, _ in dados.iterrows():
    if idx in solteira_comPais.index:
        dados['solteira_com_pais'].at[idx] = 1
        

# cols_numericas = dados.select_dtypes(include=[np.number]).columns
dados.loc[dados['solteira_com_pais']==1]['Idade'].mean() # 12.1788

dados.loc[dados['solteira_com_pais']==0]['Idade'].mean() # 31.2681
# Aqui ja vemos uma diferença absurda! Com isso temos certeza que as flags ajudarão ao modelo



