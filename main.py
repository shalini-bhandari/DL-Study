import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X = np.arange(-100, 100, 4)
y = 2 * X + 1

X_train = X.reshape(-1, 1)
y_train = y.reshape(-1, 1)

print("Sample X_train:", X_train[:5])
print("Sample y_train:", y_train[:5])

# Define the model using Sequential API
model = keras.Sequential([
    #Input layer
    layers.Input(shape = (1, )),

    #Hidden layer
    layers.Dense(units = 32, activation = 'relu'),
    layers.Dense(units = 16, activation = 'relu'),

    #Output layer
    layers.Dense(units = 1)
])

#model summary
model.summary()

# Compile the model
model.compile(
    loss = 'mean_squared_error',
    optimizer = keras.optimizers.Adam(learning_rate = 0.01),
    metrics = ['mae']
)

# Train the model
print("Starting training...")
hsitory = model.fit(
    X_train,
    y_train,
    epochs = 100,
    verbose = 1
)
print("Training completed.")

# Evaluate the model
test_value = np.array([[10.0]])
prediction = model.predict(test_value)

print(f"\nModel prediction for x = 10 : {prediction[0][0]: .2f}")
print(f"Actual value for x = 10 : {2 * 10 + 1}")