import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = {cls: len(y[y == cls]) / len(y) for cls in self.classes}

        # Initialize feature probabilities
        self.feature_probs = {cls: {} for cls in self.classes}

        # Calculate feature probabilities
        for cls in self.classes:
            X_cls = X[y == cls]
            for col in X.columns:
                values, counts = np.unique(X_cls[col], return_counts=True)
                prob_dict = {val: count / len(X_cls) for val, count in zip(values, counts)}
                self.feature_probs[cls][col] = prob_dict

    def calculate_likelihood(self, X, cls):
        """ Calculate P(X|Ci) """
        likelihood = 1
        for col, val in X.items():
            prob_dict = self.feature_probs[cls].get(col, {})
            # Apply Laplace Smoothing
            prob = prob_dict.get(val, 1e-9)
            likelihood *= prob
        return likelihood

    def calculate_posterior(self, X):
        """ Calculate P(Ci|X) for each class """
        posteriors = {}
        for cls in self.classes:
            prior = self.class_priors[cls]
            likelihood = self.calculate_likelihood(X, cls)
            posteriors[cls] = prior * likelihood
        return posteriors

    def predict(self, X):
        """ Predict the class label for each instance """
        predictions = []
        for _, row in X.iterrows():
            posteriors = self.calculate_posterior(row)
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        return predictions

# Load dataset
data = pd.read_csv('DataMining/weather_data.csv')

# Prepare features and target variable
X = data.drop(columns=['Id', 'play'])
y = data['play']

# Initialize and train the model
nb = NaiveBayes()
nb.fit(X, y)

# Example Test Data
test_data = pd.DataFrame({
    "outlook": ["sunny", "rainy", "overcast"],
    "terrain": ["flat", "slope", "undulating"],
    "temperature": ["mild", "hot", "cool"],
    "humidity": ["high", "low", "normal"],
    "wind": ["weak", "strong", "low"]
})

# Predict
predictions = nb.predict(test_data)
print("Predictions:", predictions)
