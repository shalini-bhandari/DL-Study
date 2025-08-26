import joblib
import numpy as np
from tensorflow import keras

loaded_model = keras.models.load_model("titanic_model.keras")
loaded_scaler = joblib.load("titanic_scaler.joblib")

print("Model and scaler loaded successfully.")

feature_names = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "FamilySize","IsAlone", "Embarked_Q", "Embarked_S",
    "Title_Miss", "Title_Mr", "Title_Mrs"
]

# Example usage (assuming you have new data to predict)
new_passenger = np.array([[
    3, 1, 30, 0, 0, 8.05, 1, 1, 0, 1, 0, 1, 0, 0
]])
scaled_passenger = loaded_scaler.transform(new_passenger)
prediction_prob = loaded_model.predict(scaled_passenger)
prediction = (prediction_prob > 0.5).astype("int32").flatten()[0]

print("\nNew passenger predication")
print(f"Prediction probability of survival: {prediction_prob[0][0]:.4f}")

if(prediction == 1):
    print("The passenger is predicted to Survive.")
else:
    print("The passenger is predicted to Not Survive.")