from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class XGBoostImpl:
    def __init__(self):
        #Importando o dataset de vinhos
        dataset = pd.read_csv('winequality-red.csv', sep=';')
        #carregando as colunas importantes 
        x = dataset.iloc[:, 0:11]
        #valores do target
        y = dataset.iloc[:, 11].values

        ##Como os valores do dataset são numéricos, não é necessário tratá-los 
        # ex: dados categóricos precisam ser transformados para usar o XGBoost

        # Train test split divide nosso dataset e retorna nas variáveis x_train, x_test, y_train e y_test
        # O test_size define a porcentagem de dados que serão utilizados para o teste
        # Nesse caso: 20% teste e 80% treino
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)


        # DMatrix é uma estrutura de dados utilizadas pelo XGBoost para melhorar a performance, então, faremos a conversão através do xgb.DMatrix
        # DMatrix para treino
        xgb_train = xgb.DMatrix(x_train, y_train)


        # DMatrix para teste
        xgb_test = xgb.DMatrix(x_test, y_test)

        #Parâmetros
        # reg:squarederror -> o target é um valor contínuo (exmemplo: valores de 0 até 10)
        # max_depth -> profundidade máxima da arvore
        # learning_rate -> passo do aprendizado

        param = {
            'objective' : 'reg:squarederror',
            'max_depth' : 5,
            'learning_rate' : 0.1,
        }

        # num_boost_round -> número de boosting
        model = xgb.train(params=param, dtrain=xgb_train, num_boost_round=50)

        #Vendo o que o modelo prediz com o DMatrix_test
        y_pred = model.predict(xgb_test)
        y_pred = np.round(y_pred)
        acc = accuracy_score(y_test, y_pred)
        print("Acurácia do modelo XGBoost:", acc*100,"%")

        plt.figure(figsize=(5,4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Matriz de confusão XGBoost")
        plt.xlabel("Valor predito")
        plt.ylabel("Valor real")
        plt.show()