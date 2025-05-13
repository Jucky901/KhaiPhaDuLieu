import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=3, feature_names=None):
        self.max_depth = max_depth
        self.root = None
        self.feature_names = feature_names

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # If only one class left or reached max depth, return leaf node
        if len(unique_labels) == 1 or depth == self.max_depth:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_indices = X[:, best_feature] == best_threshold
        right_indices = X[:, best_feature] != best_threshold

        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(best_feature, best_threshold, left_child, right_child)

    def _best_split(self, X, y):
        best_gain = float('-inf')
        best_feature, best_threshold = None, None
        num_samples, num_features = X.shape

        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                gain = self._information_gain(X[:, feature], y, value)
                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = feature, value

        return best_feature, best_threshold

    def _information_gain(self, feature_column, y, value):
        parent_entropy = self._entropy(y)
        left_indices = feature_column == value
        right_indices = feature_column != value
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        left_weight = len(y[left_indices]) / len(y)
        right_weight = len(y[right_indices]) / len(y)

        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _entropy(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def _most_common_label(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        most_common_index = counts.argmax()
        return unique_labels[most_common_index]

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] == node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        prefix = "    " * depth
        tree_str = ""

        if node.value is not None:
            tree_str += f"{prefix}Leaf: {node.value}\n"
        else:
            # Get the feature name instead of the index
            feature_name = self.feature_names[node.feature] if self.feature_names else f"Feature {node.feature}"
            tree_str += f"{prefix}[{feature_name} == {node.threshold}]\n"
            tree_str += self.print_tree(node.left, depth + 1)
            tree_str += self.print_tree(node.right, depth + 1)

        return tree_str

    def get_tree(self, node=None):
        """ Return the decision tree as a nested dictionary with feature names. """
        if node is None:
            node = self.root

        # Leaf Node
        if node.value is not None:
            return {"leaf": node.value}

        # Decision Node
        feature_name = self.feature_names[node.feature] if self.feature_names else f"Feature {node.feature}"
        return {
            "feature": feature_name,
            "threshold": node.threshold,
            "true": self.get_tree(node.left),
            "false": self.get_tree(node.right)
        }

