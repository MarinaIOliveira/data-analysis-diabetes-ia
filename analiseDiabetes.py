import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

#Carregamento dos dados

dados = pd.read_csv('./diabetes.txt')

#Renomeia as colunas do data frame
#colunas = ['Numero vezes grávida', 'Glicose', 'Pressão arterial', 'Espessura da dobra da pele', 'Insulina sérica','IMC','Função de Pedigree diabetes','Idade','Teste para diabetes']

#Criação data frame diabetes
diabetes_df = pd.DataFrame(dados)

#Renomeia as colunas do data frame
diabetes_df.columns = ['Numero vezes grávida', 'Glicose', 'Pressão arterial', 'Espessura da dobra da pele', 'Insulina sérica','IMC','Função de Pedigree diabetes','Idade','Teste para diabetes']

#Lista Top 5 dados do data set
diabetes_df.head()

#verificando as opções da variável resposta Possui diabes
#Identificamos que realmente há apenas duas possíveis opções de resposta ("tested_negative","tested_positive")

print(diabetes_df["Teste para diabetes"].unique())

#Substituição valores tested_negative e tested_positive para valores binários (0 e 1).

df_diabetes_novo = diabetes_df.replace({'Teste para diabetes': {"tested_negative": 0, "tested_positive": 1}})

#Lista Top 5 dados para verificar as alterações

df_diabetes_novo.head().head()

#Verificacao valores de tipos de dados e existencia de valores nulos
#Identificamos que não há valores nulos

df_diabetes_novo.info()

#Estatística descritiva sobre a base de dados

#Estatisticas descritivas

df_diabetes_novo.describe()

#Algumas observações do conjunto de dados:
#Verificamos que não existem valores ausentes no conjunto de dados.
#O numero máximo de vezes que uma pessoa engravidou foi 17.
#A pessoa com maior idade possui 81 anos enquanto a de menor idade possui 21 anos

#Verificação da variável resposta Teste para diabetes
#Verificamos que do total de 767 pessoas, 500 tiveram o teste negativo para o diabetes e 267 teste possitivo para o diabetes

df_teste_diabetes = df_diabetes_novo.groupby(["Teste para diabetes"]).size().reset_index(name='Qtd')

#renomeia as colunas 
df_teste_diabetes.columns = ["Teste para diabetes","Qtd"]

df_grafico_diabetes = df_teste_diabetes.replace({'Teste para diabetes': {0: "Teste negativo", 1: "Teste positivo"}})

plt.bar(df_grafico_diabetes["Teste para diabetes"], df_grafico_diabetes["Qtd"], width = 0.4, color='green', align='center')

plt.title('Quantidade de diabéticos na base') 
plt.xlabel('Teste para diabetes')
plt.ylabel('Quantidade')
plt.show()

#Separação dos dados para Predição
#Armazena os dados da variável resposta "Teste para diabetes" na variável y_train 
le = LabelEncoder()
y_train = le.fit_transform(df_diabetes_novo.iloc[:,(df_diabetes_novo.shape[1] - 1)])

#Cria um novo data frame sem a variável resposta que será utilizado no treinamento do modelo 

X_dict = df_diabetes_novo.iloc[:,0:8].T.to_dict().values()

vect = DictVectorizer(sparse=False)

X_train = vect.fit_transform(X_dict)

#Árvore de decisão
diabetes_tree = DecisionTreeClassifier(random_state=0)
diabetes_tree = diabetes_tree.fit(X_train, y_train)
print("Acurácia:", diabetes_tree.score(X_train, y_train))

Train_predict = diabetes_tree.predict(X_train)
print("Acurácia de previsão:", accuracy_score(y_train, Train_predict))
print(classification_report(y_train, Train_predict))

with open("diabetes_tree.dot", 'w') as f:
     f = tree.export_graphviz(diabetes_tree, out_file=f,
                              feature_names=vect.feature_names_, 
                              class_names=["Teste para diabetes=Nao", "Teste para diabetes=Sim"])


#Naive Bayes
nb_nominal = BernoulliNB()
#Aplica o algoritmo de naive bayes
nb_nominal = nb_nominal.fit(X_train, y_train)
# print da acuracia do algoritmo
print("Acurácia:", nb_nominal.score(X_train, y_train))

#cria um y_pred para o modelo Naive Bayes (Outro nome para não sobrepor o criado na arvore de decisão)
y_pred_BNB = nb_nominal.predict(X_train)
#print da acurácia da previsão
print("Acurácia de previsão:", accuracy_score(y_train, y_pred_BNB))
print(classification_report(y_train, y_pred_BNB))

