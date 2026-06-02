import numpy as np
from collections import Counter

class Node:
    '''
    Classe que representa um nó em uma árvore de decisão

    feature     -> qual variável é usada
    threshold   -> qual valor para a decisão 
    left        -> filho a esquerda
    right       -> filho a direita
    value       -> classe prevista (não será None apenas quando for folha)
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    '''
    Verifica se a classe Node é uma folha

    Retorna True se contem algum valor  -> é folha
    Retorna False se não contem valor   -> não é folha
    '''
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    '''
    Classe que constrói a árvore de decisão

    min_samples_split   -> quantidade mínima de amostras para continuar dividindo
    max_depth           -> profundidade máxima da árvore
    n_features          -> quantidade de atributos considerados para cada divisão
    root                -> aponta para a raiz da árvore
    '''
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    '''
    Método que treina a árvore
    '''
    def fit(self, X, y):
        # Usa todas as colunas caso a quantidade de atributos não tenha sido definida
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)

        # Constrói a árvore
        self.root = self._grow_tree(X, y)
    
    '''
    Método que constrói a árvore
    '''
    def _grow_tree(self, X, y, depth=0):
        # Quantidade de linhas (amostras) e colunas (atributos) em X
        n_samples, n_feats = X.shape

        # Quantidade de classes
        n_labels = len(np.unique(y))

        # Critério de parada
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            # Valor da folha (qual classe)
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Escolha das variáveis
        feat_idxs = np.random.choice(
            n_feats,
            self.n_features,
            replace=False
        )

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)
    
    '''
    Método que retorna o valor da folha
    '''
    def _most_common_label(self, y):
        # Dicionário que contem a frequência de cada item
        counter = Counter(y)

        # retorna o valor com maior frequência
        most_common_list = counter.most_common(1)
        value = most_common_list[0][0] if most_common_list else None
        return value
    
    '''
    Método que retorna a melhor variável
    '''
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        # Percorrer por todas as variáveis e encontrar a melhor escolha
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Calculo do ganho de informação
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    '''
    Método que cálcula o ganho de informação
    '''
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    '''
    Método responsável pelo cálculo da entropia
    '''
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    '''
    
    '''
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    '''
    Método para percorrer a árvore
    '''
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
