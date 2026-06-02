import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from decision_tree import DecisionTree
from random_forest import RandomForest

if __name__ == '__main__':

    data = pd.read_csv('winequality-red.csv', sep=';')

    X = data.drop(columns=['quality']).values
    y = data['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree()
    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)
    
    acuracia_arvore_decisao = accuracy_score(y_test, prediction)
    print(f'Acurácia - Árvore de Decisão: {acuracia_arvore_decisao}')

    clf = RandomForest(n_trees=20)
    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)

    acuracia_random_forest = accuracy_score(y_test, prediction)
    print(f'Acurácia - Random Forest: {acuracia_random_forest}')