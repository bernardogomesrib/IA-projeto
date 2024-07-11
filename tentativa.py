#! pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_squared_error;
from pandas_profiling import ydata_profiling;
import plotly.graph_objects as go;

link = 'https://raw.githubusercontent.com/bernardogomesrib/IA-projeto/main/archive/Student_Dropout_rate.csv';
dataset = pd.read_csv(link,sep=";")
dicionario ={"Graduate":0,"Dropout":1,"Enrolled":2}#dicionario para transformar as strings em números
#transformando as strings em números

for i in range(len(dataset['Target'].values)):
  dataset['Target'].values[i] = dicionario[dataset['Target'].values[i]]
#fim transformação

#apagando todas as linhas que tem a coluna 'Target' com valor igual a 2

dataset = dataset[dataset['Target'] != 2]
#fim apagar linhas


#transformando o dataframe em um dataframe do pandas
data = pd.DataFrame(data=dataset)

#definindo o que vai ser resultado e o que é informação
y = dataset['Target'] #resultado
x = dataset.drop(['Target'],axis=1); #informação



#dividindo os dados em teste e resultado para treino e teste em 80% treino e 20% teste 
var_treinamento,var_teste,resp_treinamento,resp_test = train_test_split(x,y,test_size = 0.2,random_state = 5)
#definindo um dataframe para visualizar como foi predito e como foi o resultado
dataframe_results = pd.DataFrame()



#definindo os valores reais no dataframe
dataframe_results['Valor_real'] = resp_test.values
#fim definição inicial do dataframe

#treinando o modelo
lin_model = LinearRegression()
lin_model.fit(var_treinamento,resp_treinamento)
#fim treinamento
predicao = lin_model.predict(var_teste)



#enviando para o dataframe 
dataframe_results['valor_predito'] = predicao
rmse = (np.sqrt(mean_squared_error(resp_test,predicao)))

print("rmse = ",rmse);



#dataframe_results

#Linha com valores reais
fig = go.Figure()
fig.add_trace(go.Scatter(x=dataframe_results.index,
                         y=dataframe_results.Valor_real,
                            mode='lines+markers',
                            name='Valor real',
                            line=dict(color='RED', width=2)))


#Linha com dados preditos
fig.add_trace(go.Scatter(x=dataframe_results.index,
                         y=dataframe_results.valor_predito,
                         mode='lines+markers',
                         name='Valor predito',
                         line=dict(color='BLUE', width=2)))
fig.show()


#usando random random forest classifier para prever os valores
from sklearn.ensemble import RandomForestClassifier
#treinando o modelo
random_forest = RandomForestClassifier(random_state=5)
#fim treinamento
random_forest.fit(var_treinamento,resp_treinamento)

#fazendo a predição
predicao = random_forest.predict(var_teste)

dataframe_results['valor_predito'] = predicao

fig = go.Figure()
fig.add_trace(go.Scatter(x=dataframe_results.index,
                         y=dataframe_results.Valor_real,
                            mode='lines+markers',
                            name='Valor real',
                            line=dict(color='RED', width=2)))


#Linha com dados preditos
fig.add_trace(go.Scatter(x=dataframe_results.index,
                         y=dataframe_results.valor_predito,
                         mode='lines+markers',
                         name='Valor predito',
                         line=dict(color='BLUE', width=2)))
fig.show()
