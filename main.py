
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    from decision_tree import DecisionTree

    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    data = pd.read_csv('winequality-red.csv', sep=';')

    X = data.drop(columns=['quality']).values
    y = data['quality'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree()
    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)
    
    acc = accuracy(y_test, prediction)
    print(acc)