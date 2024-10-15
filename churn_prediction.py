# churn_prediction.py

import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('F:\Projects\Churn_prediction\Churn.csv')

# Preprocess the data
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the first few rows of the training labels
print("Training Labels:\n", y_train.head())

# Build the neural network model
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32)

# Make predictions on the test set
y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_hat)
print("Accuracy Score:", accuracy)

# Save the model
model.save('tfmodel.keras')

# Clean up by deleting the model from memory
del model 

# Load the model again (if needed)
model = load_model('tfmodel.keras')
