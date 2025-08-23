import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print("First 5 rows of the dataset:")
print(df.head())

#Exploratory Data Analysis 
print("\nDataset information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

#Visualizations
df.hist(bins=30, figsize=(15,10))
plt.suptitle("Feature Distributions Histograms", fontsize=16)
plt.show()

# Calculate correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

X = df.drop('MedHouseVal', axis = 1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_features = X.shape[1]

# Define the model using Sequential API
model = keras.Sequential([
    #Input layer
    layers.Input(shape = (n_features,)),

    #Hidden layer
    layers.Dense(units = 64, activation = 'relu'),
    layers.Dense(units = 32, activation = 'relu'),
    layers.Dense(1)

])

# Compile the model
model.compile(
    loss = 'mean_squared_error',
    optimizer = keras.optimizers.Adam(learning_rate = 0.01),
    metrics = ['mae']
)
model.summary()

# Train the model
print("Starting training...")
hsitory = model.fit(
    X_train_scaled,
    y_train,
    epochs = 50,
    verbose = 0,
    validation_data = (X_test_scaled, y_test)
)
print("Training completed.")

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(hsitory.history['loss'], label='Training Loss')
plt.plot(hsitory.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model on test data
loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Loss (MSE): {loss[0]:.4f}")

y_pred = model.predict(X_test_scaled).flatten()

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0, 5], [0, 5], '--', color='red', lw = 2, label = 'Ideal Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.axis('equal')
plt.axis('square')
plt.legend()
plt.grid(True)
plt.show()

def predict_from_user_input(model, scaler):
    feature_names = [
        "Median Income", "House Age", "Average Rooms", "Average Bedrooms",
        "Block group population", "Average Occupancy", "Latitude", "Longitude"
    ]
    
    user_input = []
    print("Please enter the following features:")

    for feature in feature_names:
        while True:
            try:
                value = float(input(f"Enter value for {feature}: "))
                user_input.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
        
    user_input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(user_input_array)

    predicted_value = model.predict(scaled_input)
    predicted_price = predicted_value[0][0] * 100000

    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    mae_in_inr = mae * 100000

    print(f"\nPredicted Median House Value: {predicted_price:.2f}")
  
predict_from_user_input(model, scaler)