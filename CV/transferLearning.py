import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, datasets
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
X_train, X_test = X_train / 255.0, X_test / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

# load pre-trained model(VGG16)
IMG_SIZE = (75, 75)
base_model = keras.applications.VGG16(
    input_shape = IMG_SIZE + (3,),
    include_top = False,
    weights = 'imagenet'
)
# freeze the base model
base_model.trainable = False

base_model.summary()

# build the model
model = keras.Sequential([
    layers.Input(shape = (32, 32 , 3)),
    layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])
model.summary()

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs = 10,
    validation_data = (X_test, y_test)
)

# Evaluate the model
print("\nEvaluate the model:")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Visualize some predictions
predictions = model.predict(X_test)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i])
    plt.xlabel(f"Pred: {class_names[np.argmax(predictions[i])]}, True: {class_names[y_test[i][0]]}")
plt.show()
