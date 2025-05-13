import pandas as pd
from collections import defaultdict


#   NHÓM 2
#Trương Quốc Vương - 22110460 - 50%
#Nguyễn Đức Tín - 22110434 - 50% 

# Load dữ liệu
def load_data(file_path):
    return pd.read_csv(file_path)

# Tính P(Ci)
def calculate_class_probs(df, target_col):
    total = len(df)
    return {cls: count / total for cls, count in df[target_col].value_counts().items()}


# Tính P(X|Ci) nhưng chỉ với feature xuất hiện trong X_test
def calculate_conditional_probs(df, target_col, X_test):
    conditional_probs = defaultdict(dict)

    for cls in df[target_col].unique():
        subset = df[df[target_col] == cls]
        total = len(subset)
        for feature in X_test:
            value = X_test[feature]
            count = len(subset[subset[feature] == value])
            prob = count / total if total > 0 else 0
            conditional_probs[cls][(feature, value)] = prob

    return conditional_probs

# Dự đoán cho input X
def predict(X, class_probs, conditional_probs, target_col):
    results = {}
    for cls in class_probs:
        prob = class_probs[cls]
        print(f"\nP(X|{target_col} = {cls}):")
        for feature, value in X.items():
            cond_prob = conditional_probs[cls].get((feature, value), 0)
            print(f"P({feature} = {value} | {target_col} = {cls}) = {cond_prob:.3f}")
            prob *= cond_prob
        results[cls] = prob
        print(f"--> P(X|{cls}) * P({cls}) = {results[cls]:.5f}")
    predicted_class = max(results, key=results.get)
    return predicted_class, results

# Main
def main(file_path, test_inputs):
    df = load_data(file_path)
    print(df)
    target_col = 'play'
    features = [col for col in df.columns if col not in ['Id', target_col]]

    for i, row in test_inputs.iterrows():
        X_test = row.to_dict()
        print(f"\n========== Test Case {i+1} ==========")
        print(f"Test Input: {X_test}")

        # Tính xác suất
        class_probs = calculate_class_probs(df, target_col)
        print("\nP(Ci):")
        for cls, prob in class_probs.items():
            print(f"P({target_col} = {cls}) = {prob:.3f}")

        conditional_probs = calculate_conditional_probs(df, target_col, X_test)

        # Dự đoán
        prediction, probs = predict(X_test, class_probs, conditional_probs, target_col)
        print(f"\n➡️ Prediction: {target_col} = {prediction}")

        # Thêm kết quả vào dataframe tạm thời (giả lập)
        new_row = {**X_test, target_col: prediction}
        for col in df.columns:
            if col not in new_row:
                new_row[col] = None  # Thêm giá trị trống cho các cột còn lại
        df = df._append(new_row, ignore_index=True)

# =============================
# Danh sách các bộ test liên tiếp
test_inputs = pd.DataFrame({
    "outlook": ["sunny", "rainy", "overcast"],
    "terrain": ["flat", "slope", "undulating"],
    "temperature": ["mild", "cool", "hot"],
    "humidity": ["high", "normal", "low"],
    "wind": ["weak", "strong", "low"]
})

main('DataMining/weather.csv', test_inputs)
