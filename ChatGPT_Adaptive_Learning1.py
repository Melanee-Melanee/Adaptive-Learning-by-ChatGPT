from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Training data
X_train = [[0, 0], [1, 1], [1, 0], [0, 1]]
y_train = [0, 1, 1, 0]

# Initialize the model
clf = Perceptron(tol=1e-3)

# Train the model on the training data
clf.fit(X_train, y_train)

# Test data
X_test = [[1, 1], [0, 0]]
y_test = [1, 0]

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Update the model with new data
X_update = [[0, 1], [1, 0], [1, 1]]
y_update = [1, 1, 0]
clf.partial_fit(X_update, y_update)

# Make predictions on the updated data
y_pred = clf.predict(X_test)

# Calculate the updated accuracy
acc = accuracy_score(y_test, y_pred)
print("Updated accuracy:", acc)
