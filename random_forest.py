from decision_tree import DecisionTree

import numpy as np
from collections import Counter

class RandomForest:
    '''
    Classe que constrói a floresta

    n_trees             -> núemro de árvores na floresta
    max_depth           -> número máximo de níveis em cada árvore de decisão
    min_samples_split   -> número mínimo de amostras
    feature             -> qual variável é usada

    trees               -> árvores
    '''
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature = feature

        self.trees = []
    
    '''
    Método que treina a árvore
    '''
    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                n_features = self.feature
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(
            n_samples,
            n_samples,
            replace=True
        )
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common_list = counter.most_common(1)
        most_common = most_common_list[0][0] if most_common_list else None
        return most_common
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_predictions])
        return predictions