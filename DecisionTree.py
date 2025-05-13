import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        if len(unique_labels) == 1 or depth == self.max_depth:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y)
        left_indices = X[:, best_feature] == best_threshold
        right_indices = X[:, best_feature] != best_threshold

        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(best_feature, best_threshold, left_child, right_child)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        num_samples, num_features = X.shape

        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                gini = self._gini_index(X[:, feature], y, value)
                if gini < best_gini:
                    best_gini = gini
                    best_feature, best_threshold = feature, value

        return best_feature, best_threshold

    def _gini_index(self, feature_column, y, value):
        left_indices = feature_column == value
        right_indices = feature_column != value
        left_gini = self._gini(y[left_indices])
        right_gini = self._gini(y[right_indices])

        left_weight = len(y[left_indices]) / len(y)
        right_weight = len(y[right_indices]) / len(y)

        return left_weight * left_gini + right_weight * right_gini

    def _gini(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def _most_common_label(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        most_common_index = counts.argmax()
        return unique_labels[most_common_index]

    def print_tree(self, node=None, depth=0, branch="Root"):
        if node is None:
            node = self.root

        # If this is a leaf node
        if node.value is not None:
            print("  " * depth + f"{branch} Leaf: {node.value}")
            return

        # Otherwise, print the current node
        print("  " * depth + f"[[Feature {node.feature} == {node.threshold}]]")

        # Recursively print left and right branches
        self.print_tree(node.left, depth + 2, branch="Left")
        self.print_tree(node.right, depth + 2, branch="Right")

    def _get_tree_string(self, node, depth, branch="Root"):
        """ Helper method to generate the tree structure as a string with left/right indication """
        if node is None:
            return "  " * depth + "None"

        # If this is a leaf node
        if node.value is not None:
            return "  " * depth + f"{branch} Leaf: {node.value}"

        # Otherwise, print the current node
        tree_string = "  " * depth + f"[[Feature {node.feature} == {node.threshold}]]\n"
        left_tree = self._get_tree_string(node.left, depth + 2, branch="Left")
        right_tree = self._get_tree_string(node.right, depth + 2, branch="Right")

        tree_string += "  " * (depth + 1) + "/               \\\n"
        tree_string += "  " * (depth + 1) + "Left:          Right:\n"
        tree_string += left_tree + "\n" + right_tree
        return tree_string


    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] == node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

# Example usage
if __name__ == "__main__":
    data = pd.read_csv('DataMining/weather_data.csv')
    X = data.drop(columns=['Id','play']).values
    y = data['play'].values

    model = DecisionTree(max_depth=3)
    model.fit(X, y)

        # Print the tree structure
    print("Decision Tree Structure:")
    model.print_tree()
