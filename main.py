import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from decision_tree import DecisionTree
from xgboost_impl import XGBoostImpl
from random_forest import RandomForest

if __name__ == '__main__':

    data = pd.read_csv('winequality-red.csv', sep=';')

    X = data.drop(columns=['quality']).values
    y = data['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    # Decision Tree
    clf = DecisionTree()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acuracia_arvore_decisao = accuracy_score(y_test, y_pred)
    print(f'Acurácia - Decision Tree: {acuracia_arvore_decisao}')

    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de confusão - Decision Tree')
    plt.xlabel('Valor predito')
    plt.ylabel('Valor real')
    plt.show()

    # Random Forest
    clf = RandomForest(n_trees=20)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acuracia_random_forest = accuracy_score(y_test, y_pred)
    print(f'Acurácia - Random Forest: {acuracia_random_forest}')

    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de confusão - Random Forest')
    plt.xlabel('Valor predito')
    plt.ylabel('Valor real')
    plt.show()

    # XGBoost
    XGBoostImpl()
