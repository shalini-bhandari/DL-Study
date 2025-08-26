import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np, tensorflow as tf, random, os
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)



titanic = pd.read_csv('titanic_data.csv')
print("Initial data:")
print(titanic.head())
print(titanic.info())

# Clean the data
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Feature Engineering
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1
titanic['IsAlone'] = (titanic['FamilySize'] == 1).astype(int)

titanic['Title'] = titanic['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles
titanic['Title'] = titanic['Title'].replace(['Mlle','Ms'],'Miss')
titanic['Title'] = titanic['Title'].replace('Mme','Mrs')
rare_titles = titanic['Title'].value_counts()[titanic['Title'].value_counts() < 10].index
titanic['Title'] = titanic['Title'].replace(rare_titles, 'Rare')

titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic = pd.get_dummies(titanic, columns=['Embarked', 'Title'], drop_first=True)
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')

print("\nCleaned + Engineered data:")
print(titanic.head())


X = titanic.drop('Survived', axis=1)
y = titanic['Survived'].astype("int32")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

# Build the model
model = Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    Dropout(0.3),
    layers.Dense(32, activation='relu'),
    Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# compile the model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer = optimizer,
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=200,          # higher, but training will stop early
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# evaluate the model
val_accuracy = history.history['val_accuracy'][-1]
print(f'\nValidation Accuracy: {val_accuracy * 100 :.2f}%')