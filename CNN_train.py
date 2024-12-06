import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import sys
import numpy as np

if __name__ == "__main__":

    # Parameters

    input_size = (96, 96)
    batch_size = 64
    num_classes = 5
    epochs = 15

    boosting = False

    train_dir = "dataset/train"
    test_dir = "dataset/test"

    loss_fn = "categorical_crossentropy"
 
    if boosting:
        train_dir = "dataset/train_boosted"

    # Data Augmentation and Preprocessing
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)

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

    _lambda = 10e-5 # L2 Regularization
    # Build the CNN Model with L2 regularization and more dense layers
    model = Sequential([
        Conv2D(64, (3, 3), activation="relu", input_shape=(96, 96, 3), kernel_regularizer=l2(_lambda)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(_lambda)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation="relu", kernel_regularizer=l2(_lambda)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation="relu", kernel_regularizer=l2(_lambda)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.1),
        Dense(128, activation="relu", kernel_regularizer=l2(_lambda)),
        Dropout(0.1),
        Dense(128, activation="relu", kernel_regularizer=l2(_lambda)),
        Dropout(0.1),
        Dense(num_classes, activation="softmax")
    ])


    model.compile(
        optimizer="adam",
        loss=loss_fn,
        metrics=["f1_score"],
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
    if boosting:
        model.save("cnn_model_boosting.keras")
    else:
        model.save("cnn_model.keras")

    # Plot and save Loss and Accuracy
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')


    # Plot and save F1 Score if available
    f1_scores = history.history['f1_score']
    val_f1_scores = history.history['val_f1_score']
    f1_scores = np.mean(f1_scores, axis=1)
    val_f1_scores = np.mean(val_f1_scores, axis=1)
    plt.figure()
    plt.plot(f1_scores, label='F1 Score')
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig('f1_score_plot.png')

    sys.exit(0)