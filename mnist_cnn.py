from keras.callbacks import EarlyStopping
from keras import layers, models
import time
import tensorflow as tf
import random
import numpy as np

SEED = 333

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def cnn_model(x_train, y_train, x_val, y_val, x_test, y_test):
    cnn = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])

    cnn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=2,
        restore_best_weights=True
    )

    start_time = time.time()
    history = cnn.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[early_stop]
    )
    cnn_training_time = time.time() - start_time

    test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=0)
  
    print()
    print(f"CNN training time: {cnn_training_time:.2f}s")
    print(f"CNN validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"CNN test accuracy: {test_acc:.4f}")
    print()

    return cnn, history