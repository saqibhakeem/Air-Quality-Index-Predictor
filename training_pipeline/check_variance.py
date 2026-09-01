from load_features import get_train_test_split

X_train, X_test, y_train, y_test = get_train_test_split()

print(y_test.describe())
print()
print("std relative to mean:", (y_test.std() / y_test.mean() * 100).round(1), "%")