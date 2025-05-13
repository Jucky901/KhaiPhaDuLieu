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
