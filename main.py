import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X = np.arrange(-100, 100, 4)
y = 2 * X + 1

X_train = X.reshape(-1, 1)
y_train = y.reshape(-1, 1)
