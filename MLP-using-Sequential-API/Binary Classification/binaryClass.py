import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


titanic = pd.read_csv('titanic_data.csv')
print("Initial data:")
print(titanic.head())

# Clean the data
titanic.fillna({'Age': 'median'}, inplace=True)
titanic.fillna({'Fare': 'median'}, inplace=True)
titanic.fillna({'Embarked': 'mode'}, inplace=True)

titanic.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True) 
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic = pd.get_dummies(titanic, columns=['Embarked'], drop_first=True)
titanic.drop(['Name', 'Ticket', 'PassengerId', 'Fare'], axis=1, inplace=True, errors='ignore')

print("\nCleaned data:")
print(titanic.head())

# Split the data into features and target
X = titanic.drop('Survived', axis = 1)
y = titanic['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state = 42
    )
print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

# Build the model
model = keras.Sequential([
    layers.Input(shape = (X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    # output layer for binary classification
    layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs = 100,
    batch_size = 32,
    validation_split = 0.2,
    verbose = 0
)

# evaluate the model
#val_accuracy = history.history['val_accuracy'][-1]
#print(f'\nValidation Accuracy: {val_accuracy * 100 :.2f}%')
