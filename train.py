import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import sys

# Parameters
input_size = (96, 96)
batch_size = 64
num_classes = 5
epochs = 30

boosting = True

train_dir = "dataset/train"
test_dir = "dataset/test"

if boosting:
    train_dir = "dataset/train_boosted"

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode="categorical"
)

_lambda = 10e-6
# Build the CNN Model with L2 regularization and more dense layers
model = Sequential([
    Conv2D(64, (3, 3), activation="relu", input_shape=(96, 96, 3), kernel_regularizer=l2(_lambda)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(_lambda)),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation="relu", kernel_regularizer=l2(_lambda)),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(_lambda)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu", kernel_regularizer=l2(_lambda)),
    #Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", "f1_score"]
)

# Model Summary
model.summary()

# Train the Model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# Save the Model
model.save("cnn_model.keras")

sys.exit(0)