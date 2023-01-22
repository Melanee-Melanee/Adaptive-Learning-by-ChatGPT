from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize the model
clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Update the model with new data
X_update, y_update = make_classification(n_samples=200, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

clf.fit(X_update, y_update)

# Make predictions on the updated data
y_pred = clf.predict(X_test)

# Calculate the updated accuracy
acc = accuracy_score(y_test, y_pred)
print("Updated accuracy:", acc)