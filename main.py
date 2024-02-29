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

test.shape, train.shape # Diferença entre, train é para treinar o modelo, consequentemente mais colunas e linhas.

test['Survived'] = np.nan

dados = pd.concat([train, test], sort=False)
dados.shape

dados.columns = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Cabin', 'Embarked']